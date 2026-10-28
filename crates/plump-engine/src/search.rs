//! Belief-guided determinized search (PIMC) and an endgame solver.
//!
//! Plump is imperfect information: a player sees their own hand plus all
//! played cards, but not the opponents' hands (except in the 1-card
//! Pannplump round, where the situation is reversed). The search therefore
//! samples worlds that are consistent with the viewer's information and
//! solves each determinized world with a fast MaxN search.

use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

use crate::engine::{rank_of, suit_of, terminal_reward, FastEnv, Phase, FULL_DECK};

/// Rough per-root-action node budget. Exact solving of the last tricks of a
/// four-player game has an enormous branching factor; the minimum budgets
/// below keep interactive play responsive while giving the important
/// endgame branches a real share of work. Nodes that still run out are
/// finished by the heuristic rollout.
pub const SEARCH_NODE_BUDGET: usize = 6_000;
/// Focused node budget for the first two tricks of MaxN lookahead.
pub const EARLY_NODE_BUDGET: usize = 2_000;
/// Minimum focused budget when at most four tricks remain.
pub const ENDGAME_4_NODE_BUDGET: usize = 4_000;
/// Minimum focused budget when exactly five tricks remain.
pub const ENDGAME_5_NODE_BUDGET: usize = 8_000;
/// MaxN lookahead depth after a root move in the early game.
pub const EARLY_LOOKAHEAD_DEPTH: u8 = 7;

#[inline]
fn cards_from_mask(mask: u64) -> Vec<u8> {
    let mut cards = Vec::with_capacity(mask.count_ones() as usize);
    let mut m = mask;
    while m != 0 {
        cards.push(m.trailing_zeros() as u8);
        m &= m - 1;
    }
    cards
}

#[inline]
fn bids_from_mask(mask: u16) -> Vec<u8> {
    let mut bids = Vec::with_capacity(mask.count_ones() as usize);
    let mut m = mask;
    while m != 0 {
        bids.push(m.trailing_zeros() as u8);
        m &= m - 1;
    }
    bids
}

#[inline]
fn rewards(env: &FastEnv) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for p in 0..4 {
        out[p] = terminal_reward(env.tricks_won[p], env.bids[p]);
    }
    out
}

#[inline]
fn table_bits(env: &FastEnv) -> u64 {
    let mut bits = 0u64;
    for i in 0..env.table_len as usize {
        bits |= 1u64 << env.table[i].1;
    }
    bits
}

/// Heuristic card for rollouts and for the `--search 0` greedy mode.
/// Upgraded to Tactician bot so rollouts are significantly smarter.
pub fn heuristic_card(env: &FastEnv) -> u8 {
    crate::bots::tactician_card(env)
}

/// Heuristic bid used by rollouts (and as a heuristic baseline).
/// Upgraded to Tactician bot so rollouts are significantly smarter.
pub fn heuristic_bid(env: &FastEnv) -> u8 {
    crate::bots::tactician_bid(env)
}

fn rollout(mut env: FastEnv) -> [f32; 4] {
    let mut guard = 0;
    while env.phase != Phase::Terminal && guard < 128 {
        match env.phase {
            Phase::Bidding => {
                let bid = heuristic_bid(&env);
                env.step_bid(bid);
            }
            Phase::Playing => {
                let card = heuristic_card(&env);
                env.step_play(card);
            }
            Phase::Terminal => break,
        }
        guard += 1;
    }
    rewards(&env)
}

/// MaxN over a finite, well-determined world.
///
/// `budget` is consumed by value and split evenly among every legal action
/// at this node. This deliberately prevents depth-first starvation: if the
/// first branch exhausts a shared mutable budget, all shallow sibling moves
/// would be skipped. A child which receives a zero allocation finishes with
/// the heuristic rollout, but it is still evaluated.
fn solve_inner(env: &FastEnv, depth_left: u8, budget: usize) -> [f32; 4] {
    if env.phase == Phase::Terminal {
        return rewards(env);
    }

    if depth_left == 0 || budget == 0 {
        return rollout(*env);
    }
    let depth = depth_left;

    let p = env.current_player as usize;
    let player = p;

    match env.phase {
        Phase::Bidding => {
            let mask = env.legal_bids_mask(p);
            if mask == 0 {
                return rollout(*env);
            }
            let actions = bids_from_mask(mask);
            let per_child = (budget - 1) / actions.len();
            let mut best: Option<[f32; 4]> = None;
            let mut best_value = f32::NEG_INFINITY;
            for bid in actions {
                let mut next = *env;
                next.step_bid(bid);
                let value = solve_inner(&next, depth.saturating_sub(1), per_child);
                if value[player] > best_value {
                    best_value = value[player];
                    best = Some(value);
                }
            }
            best.unwrap_or_else(|| rollout(*env))
        }
        Phase::Playing => {
            let mask = env.legal_cards_mask(p);
            if mask == 0 {
                return rollout(*env);
            }
            let actions = cards_from_mask(mask);
            let per_child = (budget - 1) / actions.len();
            let mut best: Option<[f32; 4]> = None;
            let mut best_value = f32::NEG_INFINITY;
            for card in actions {
                let mut next = *env;
                next.step_play(card);
                let value = solve_inner(&next, depth.saturating_sub(1), per_child);
                if value[player] > best_value {
                    best_value = value[player];
                    best = Some(value);
                }
            }
            best.unwrap_or_else(|| rollout(*env))
        }
        Phase::Terminal => rewards(env),
    }
}

/// MaxN value vector of a fully determined world. `depth_left == 0` means
/// "evaluate with the heuristic rollout"; PIMC uses the card-policy depth/budget
/// floors above to avoid the old zero-depth fallback everywhere.
pub fn solve_world_recursive(env: &FastEnv, depth_left: u8) -> [f32; 4] {
    solve_inner(env, depth_left, SEARCH_NODE_BUDGET)
}

/// Sample `n_worlds` deals consistent with `viewer`'s information set.
///
/// * In a 1-card round the viewer sees the three opponents' cards; only the
///   viewer's own forehead card is unknown.
/// * Otherwise the unknown pool is distributed over the three opponents
///   respecting their public void masks and the optional belief
///   probabilities from the Transformer's `belief_head`.
pub fn sample_consistent_worlds(
    base: &FastEnv,
    viewer: usize,
    n_worlds: usize,
    belief_probs: Option<&[[f32; 52]; 3]>,
) -> Vec<FastEnv> {
    assert!(viewer < 4);
    if n_worlds == 0 {
        return Vec::new();
    }
    let mut rng = rand::thread_rng();
    let table = table_bits(base);

    if base.round_cards == 1 {
        let known_opponents = (0..3).fold(0u64, |acc, i| acc | base.hands[(viewer + i + 1) % 4]);
        let pool = FULL_DECK & !known_opponents & !base.played & !table;
        let cards = cards_from_mask(pool);
        if cards.is_empty() {
            return vec![*base; n_worlds];
        }
        (0..n_worlds)
            .map(|_| {
                let mut env = *base;
                let card = cards[rng.gen_range(0..cards.len())];
                env.hands[viewer] = 1u64 << card;
                env
            })
            .collect()
    } else {
        let pool = FULL_DECK & !base.hands[viewer] & !base.played & !table;
        let counts = [
            base.hands[(viewer + 1) % 4].count_ones() as usize,
            base.hands[(viewer + 2) % 4].count_ones() as usize,
            base.hands[(viewer + 3) % 4].count_ones() as usize,
        ];
        let mut result = Vec::with_capacity(n_worlds);
        for _ in 0..n_worlds {
            let mut remaining = cards_from_mask(pool);
            let mut order = [0usize, 1, 2];
            order.shuffle(&mut rng);
            let mut sampled = [0u64; 3];
            for &rel in &order {
                let abs = (viewer + rel + 1) % 4;
                let void = base.void_mask[abs];
                let weights = belief_probs.map(|b| b[rel]);
                let mut hand = 0u64;
                for _ in 0..counts[rel] {
                    if remaining.is_empty() {
                        break;
                    }
                    let idx = if let Some(w) = weights {
                        weighted_index(&remaining, &w, void, &mut rng)
                    } else {
                        uniform_index(&remaining, void, &mut rng)
                    };
                    let card = remaining.swap_remove(idx);
                    hand |= 1u64 << card;
                }
                sampled[rel] = hand;
            }
            let mut env = *base;
            for rel in 0..3 {
                let abs = (viewer + rel + 1) % 4;
                env.hands[abs] = sampled[rel];
            }
            result.push(env);
        }
        result
    }
}

fn uniform_index(pool: &[u8], void: u8, rng: &mut impl Rng) -> usize {
    let allowed: Vec<usize> = pool
        .iter()
        .enumerate()
        .filter(|(_, &c)| void & (1u8 << suit_of(c)) == 0)
        .map(|(i, _)| i)
        .collect();
    if allowed.is_empty() {
        rng.gen_range(0..pool.len())
    } else {
        allowed[rng.gen_range(0..allowed.len())]
    }
}

fn weighted_index(pool: &[u8], weights: &[f32; 52], void: u8, rng: &mut impl Rng) -> usize {
    let mut total = 0.0f64;
    let mut w = Vec::with_capacity(pool.len());
    for &card in pool {
        let suit = suit_of(card);
        let weight = if void & (1u8 << suit) != 0 {
            0.0
        } else {
            (weights[card as usize].max(0.0) as f64).max(1e-6)
        };
        total += weight;
        w.push(weight);
    }
    if total <= 0.0 {
        return uniform_index(pool, void, rng);
    }
    let mut target = rng.gen::<f64>() * total;
    for (i, weight) in w.iter().enumerate() {
        target -= weight;
        if target <= 0.0 {
            return i;
        }
    }
    pool.len() - 1
}

/// Solve a child state reached immediately after a root card move. The
/// remaining-tricks policy is applied here instead of relying on the old
/// `depth_left = 0`, which previously disabled all MaxN lookahead until the
/// very last five tricks.
fn solve_after_root_card(env: &FastEnv) -> [f32; 4] {
    let remaining = env.round_cards.saturating_sub(env.current_trick);
    if remaining > 5 {
        solve_inner(env, EARLY_LOOKAHEAD_DEPTH, EARLY_NODE_BUDGET)
    } else if remaining == 5 {
        solve_inner(env, 20, ENDGAME_5_NODE_BUDGET)
    } else {
        solve_inner(env, 16, ENDGAME_4_NODE_BUDGET)
    }
}

/// Belief-weighted PIMC card values.
///
/// Returns the average expected `viewer` terminal reward for all 52 cards;
/// illegal card slots contain `f32::NEG_INFINITY`.
pub fn search_card_values(
    base: &FastEnv,
    viewer: usize,
    n_worlds: usize,
    belief_probs: Option<&[[f32; 52]; 3]>,
) -> [f32; 52] {
    let mut values = [f32::NEG_INFINITY; 52];
    if base.phase != Phase::Playing || base.current_player as usize != viewer {
        return values;
    }
    let legal = base.legal_cards_mask(viewer);
    if legal == 0 {
        return values;
    }
    let cards = cards_from_mask(legal);
    if cards.len() == 1 {
        values[cards[0] as usize] = 0.0;
        return values;
    }

    let worlds = sample_consistent_worlds(base, viewer, n_worlds.max(1), belief_probs);
    let per_world: Vec<Vec<f32>> = worlds
        .par_iter()
        .map(|world| {
            cards
                .iter()
                .map(|&card| {
                    let mut env = *world;
                    env.current_player = viewer as u8;
                    if env.legal_cards_mask(viewer) & (1u64 << card) == 0 {
                        // The viewer's own hand is identical in every
                        // consistent world, so this should never happen.
                        return f32::NEG_INFINITY;
                    }
                    env.step_play(card);
                    solve_after_root_card(&env)[viewer]
                })
                .collect()
        })
        .collect();

    let mut sums = vec![0.0f64; cards.len()];
    for row in &per_world {
        for (i, value) in row.iter().enumerate() {
            sums[i] += *value as f64;
        }
    }
    for (&card, sum) in cards.iter().zip(&sums) {
        values[card as usize] = (sum / worlds.len() as f64) as f32;
    }
    values
}

/// Belief-weighted PIMC: evaluate every legal card over sampled worlds and
/// return the card with the highest average `viewer` terminal reward.
pub fn search_best_card(
    base: &FastEnv,
    viewer: usize,
    n_worlds: usize,
    belief_probs: Option<&[[f32; 52]; 3]>,
) -> u8 {
    if base.phase != Phase::Playing || base.current_player as usize != viewer {
        return 0;
    }
    let legal = base.legal_cards_mask(viewer);
    if legal == 0 {
        return 0;
    }

    let values = search_card_values(base, viewer, n_worlds, belief_probs);
    let mut best_card = 0u8;
    let mut best_value = f32::NEG_INFINITY;
    for card in cards_from_mask(legal) {
        let value = values[card as usize];
        if value > best_value {
            best_value = value;
            best_card = card;
        }
    }
    best_card
}

/// Simulate the single trick after all bids are known.
fn evaluate_completed_bids(mut env: FastEnv, viewer: usize) -> f32 {
    env.phase = Phase::Playing;
    env.current_player = env.get_leader() as u8;
    env.current_trick = 0;
    env.table_len = 0;
    env.led_suit = 255;
    let mut guard = 0;
    while env.phase != Phase::Terminal && guard < 8 {
        let p = env.current_player as usize;
        let mask = env.legal_cards_mask(p);
        if mask == 0 {
            break;
        }
        let card = mask.trailing_zeros() as u8;
        env.step_play(card);
        guard += 1;
    }
    terminal_reward(env.tricks_won[viewer], env.bids[viewer])
}

/// Weighted probability of a later bidder's response. A later bidder in the
/// forehead round sees the three other forehead cards (including the
/// viewer's candidate card), so a visible Queen/King/Ace is a strong signal
/// that it is safer to pass rather than gamble on a bid of one.
fn later_bid_weight(env: &FastEnv, bidder: usize, mask: u16, bid: u8) -> f64 {
    let options = bids_from_mask(mask);
    let both_two_bid_options = options.iter().any(|&b| b == 0) && options.iter().any(|&b| b == 1);
    if !both_two_bid_options {
        return 1.0 / options.len().max(1) as f64;
    }

    let mut saw_high_card = false;
    for &opp in env.hands.iter() {
        if opp == env.hands[bidder] {
            // A bidder never sees their own forehead card.
            continue;
        }
        let mut bits = opp;
        while bits != 0 {
            let card = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            if rank_of(card) >= 12 {
                saw_high_card = true;
                break;
            }
        }
        if saw_high_card {
            break;
        }
    }
    let pass_weight = if saw_high_card { 0.85 } else { 0.60 };
    match bid {
        0 => pass_weight,
        1 => 1.0 - pass_weight,
        _ => 0.0,
    }
}

fn enumerate_later_bids(
    env: FastEnv,
    viewer: usize,
    viewer_offset: usize,
    offset: usize,
    weight: f64,
    sum: &mut f64,
    count: &mut f64,
) {
    if offset == 4 {
        *sum += weight * evaluate_completed_bids(env, viewer) as f64;
        *count += weight;
        return;
    }
    if offset <= viewer_offset {
        enumerate_later_bids(env, viewer, viewer_offset, offset + 1, weight, sum, count);
        return;
    }
    let p = (env.start_bidder as usize + offset) % 4;
    let mut bidding_env = env;
    bidding_env.current_player = p as u8;
    let mask = bidding_env.legal_bids_mask(p);
    for bid in bids_from_mask(mask) {
        let branch_weight = weight * later_bid_weight(&env, p, mask, bid);
        let mut next = env;
        next.bids[p] = bid as i8;
        next.bids_made += 1;
        enumerate_later_bids(
            next,
            viewer,
            viewer_offset,
            offset + 1,
            branch_weight,
            sum,
            count,
        );
    }
}

/// Expected terminal reward of every legal bid in a 1-card forehead round.
///
/// The viewer already sees all three opponent cards, so the only unknown is
/// the viewer's own card. Every candidate own card is enumerated, and every
/// completion of the remaining bids is evaluated. Illegal bids are
/// `NEG_INFINITY`, so callers can blend these values with a model prior.
pub fn search_bid_values_pannplump(base: &FastEnv, viewer: usize, _n_samples: usize) -> [f32; 11] {
    let mut values = [f32::NEG_INFINITY; 11];
    if base.round_cards != 1 || base.phase != Phase::Bidding {
        return values;
    }
    let legal = base.legal_bids_mask(viewer);
    if legal == 0 {
        return values;
    }
    let known_opponents = (0..3).fold(0u64, |acc, i| acc | base.hands[(viewer + i + 1) % 4]);
    let pool = FULL_DECK & !known_opponents & !base.played & !table_bits(base);
    let cards = cards_from_mask(pool);
    if cards.is_empty() {
        for bid in bids_from_mask(legal) {
            values[bid as usize] = 0.0;
        }
        return values;
    }
    let viewer_offset = base.bidding_offset(viewer);

    for bid in bids_from_mask(legal) {
        let mut total = 0.0f64;
        let mut count = 0.0f64;
        for &card in &cards {
            let mut env = *base;
            env.hands[viewer] = 1u64 << card;
            env.bids[viewer] = bid as i8;
            env.bids_made += 1;
            enumerate_later_bids(env, viewer, viewer_offset, 0, 1.0, &mut total, &mut count);
        }
        values[bid as usize] = if count > 0.0 {
            (total / count) as f32
        } else {
            0.0
        };
    }
    values
}

/// Exact Monte-Carlo solver for a 1-card forehead bid.
///
/// Returns the legal bid with the highest expected terminal reward (the
/// earliest legal bid on ties).
pub fn search_best_bid_pannplump(base: &FastEnv, viewer: usize, n_samples: usize) -> u8 {
    let values = search_bid_values_pannplump(base, viewer, n_samples);
    let legal = base.legal_bids_mask(viewer);
    let mut best_bid = 0u8;
    let mut best_value = f32::NEG_INFINITY;
    for bid in bids_from_mask(legal) {
        let value = values[bid as usize];
        if value > best_value {
            best_value = value;
            best_bid = bid;
        }
    }
    best_bid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{card_index, display_points, BID_BASE, CARD_BASE, FEAT_DIM, MAX_SEQ};

    fn pannplump_env(viewer_card: u8, opp_cards: [u8; 3], start_bidder: u8) -> FastEnv {
        let mut env = FastEnv::default();
        env.round_cards = 1;
        env.start_bidder = start_bidder;
        env.current_player = start_bidder;
        env.phase = Phase::Bidding;
        env.hands[0] = 1u64 << viewer_card;
        env.hands[1] = 1u64 << opp_cards[0];
        env.hands[2] = 1u64 << opp_cards[1];
        env.hands[3] = 1u64 << opp_cards[2];
        env
    }

    #[test]
    fn sampled_pannplump_worlds_only_unknown_card_varies() {
        let env = pannplump_env(51, [10, 20, 30], 0);
        let worlds = sample_consistent_worlds(&env, 0, 64, None);
        assert_eq!(worlds.len(), 64);
        for world in worlds {
            assert_eq!(world.hands[1], env.hands[1]);
            assert_eq!(world.hands[2], env.hands[2]);
            assert_eq!(world.hands[3], env.hands[3]);
            assert_eq!(world.hands[0].count_ones(), 1);
            let card = world.hands[0].trailing_zeros() as u8;
            assert!(![10u8, 20, 30].contains(&card));
        }
    }

    #[test]
    fn sampled_worlds_respect_counts_voids_and_known_cards() {
        let mut env = FastEnv::default();
        env.round_cards = 3;
        env.start_bidder = 0;
        env.current_player = 0;
        env.phase = Phase::Playing;
        env.hands = [
            0b111,
            1u64 << card_index(2, 5) | 1u64 << card_index(2, 6) | 1u64 << card_index(3, 7),
            1u64 << card_index(1, 8) | 1u64 << card_index(1, 9) | 1u64 << card_index(0, 10),
            1u64 << card_index(0, 11) | 1u64 << card_index(0, 12) | 1u64 << card_index(1, 13),
        ];
        env.void_mask[1] = 1; // player 1 has publicly shown out of hearts
        let worlds = sample_consistent_worlds(&env, 0, 32, None);
        for world in worlds {
            assert_eq!(world.hands[0], env.hands[0]);
            assert_eq!(world.hands[1].count_ones(), 3);
            assert_eq!(world.hands[2].count_ones(), 3);
            assert_eq!(world.hands[3].count_ones(), 3);
            // Player 1 is publicly void in hearts.
            assert!(world.hands[1] & 0x1FFF == 0);
            // Hands are disjoint and exclude the viewer's known cards.
            assert_eq!(world.hands[1] & world.hands[2], 0);
            assert_eq!(world.hands[1] & world.hands[3], 0);
            assert_eq!(world.hands[2] & world.hands[3], 0);
            let all = world.hands[0] | world.hands[1] | world.hands[2] | world.hands[3];
            assert_eq!(all.count_ones(), 12);
        }
    }

    #[test]
    fn terminal_solver_returns_rewards() {
        let mut env = FastEnv::default();
        env.phase = Phase::Terminal;
        env.bids = [2, 1, 0, 3];
        env.tricks_won = [2, 0, 1, 1];
        let values = solve_world_recursive(&env, 0);
        for p in 0..4 {
            assert_eq!(values[p], terminal_reward(env.tricks_won[p], env.bids[p]));
        }
    }

    #[test]
    fn pimc_returns_a_legal_card() {
        let env = pannplump_env(51, [10, 20, 30], 0);
        // Bidding has not happened yet; finish it manually.
        let mut env = env;
        env.bids = [1, 0, 0, 0];
        env.bids_made = 4;
        env.phase = Phase::Playing;
        env.current_player = 0;
        let card = search_best_card(&env, 0, 16, None);
        assert_eq!(card, 51);
    }

    #[test]
    fn pannplump_bid_is_legal_and_sane() {
        // Viewer 0 sees A, K, Q of the other players. Whatever the viewer's
        // card, bidding 1 should not be worse than bidding 0 (they can only
        // win the trick if their unknown card is the highest of the led
        // suit; with A/K/Q out the expectation is close). Just check legality.
        let env = pannplump_env(0, [51, 50, 49], 0);
        let bid = search_best_bid_pannplump(&env, 0, 49);
        assert!(bid <= 1);
        assert!(env.legal_bids_mask(0) & (1u16 << bid) != 0);
    }

    #[test]
    fn pannplump_bid_values_mask_illegal_bids() {
        let env = pannplump_env(0, [51, 50, 49], 0);
        let values = search_bid_values_pannplump(&env, 0, 49);
        let legal = env.legal_bids_mask(0);
        for bid in 0..=1usize {
            if legal & (1u16 << bid) != 0 {
                assert!(values[bid].is_finite());
            }
        }
        assert!(values[2..].iter().all(|v| *v == f32::NEG_INFINITY));
    }

    #[test]
    fn heuristic_play_follows_suit_and_ducks() {
        let mut env = FastEnv::default();
        env.round_cards = 2;
        env.bids = [1, 0, 0, 0];
        env.tricks_won = [1, 0, 0, 0];
        env.phase = Phase::Playing;
        env.current_player = 1;
        env.table[0] = (0, card_index(0, 14));
        env.table_len = 1;
        env.led_suit = 0;
        // Player 1 has the 5H and the AS. They have made their bid (0) so
        // they should duck with the lowest legal card (both are legal since
        // they cannot follow hearts).
        env.hands[1] = (1u64 << card_index(1, 14)) | (1u64 << card_index(0, 5));
        let card = heuristic_card(&env);
        assert_eq!(card, card_index(0, 5));
    }

    #[test]
    fn encoding_helper_used_by_search_is_consistent() {
        // Sanity check that search operates on the same public history that
        // the observation encoder exposes.
        let env = pannplump_env(51, [10, 20, 30], 2);
        let mut toks = [0i64; MAX_SEQ];
        let mut feats = [0f32; FEAT_DIM];
        let mut opp = [0f32; 156];
        env.encode_decision(0, &mut toks, &mut feats, &mut opp);
        assert_eq!(toks[0], 1);
        assert_eq!(toks[1], BID_BASE as i64 + 1);
        assert_eq!(feats[52 + 10], 1.0);
        assert_eq!(opp[0 * 52 + 10], 1.0);
        assert_eq!(display_points(0, 0), 5);
        assert_eq!(CARD_BASE as i64 + 51, 68);
    }
}
