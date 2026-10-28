//! 4-tier hardcoded bot ladder:
//! 0 = Rookie (random/low)
//! 1 = Regular (basic card counter + follow/duck)
//! 2 = Tactician (boss-card memory, finesse, high off-suit sloughing, exact forehead math)
//! 3 = Final Boss (Tactician + parallel PIMC & exact MaxN endgame solver)

use rand::seq::SliceRandom;

use crate::engine::{rank_of, suit_bits, suit_of, FastEnv, Phase, FULL_DECK};
use crate::search::{search_best_bid_pannplump, search_best_card};

pub const BOT_ROOKIE: u8 = 0;
pub const BOT_REGULAR: u8 = 1;
pub const BOT_TACTICIAN: u8 = 2;
pub const BOT_FINAL_BOSS: u8 = 3;

#[inline]
pub fn cards_from_mask(mask: u64) -> Vec<u8> {
    let mut cards = Vec::with_capacity(mask.count_ones() as usize);
    let mut m = mask;
    while m != 0 {
        cards.push(m.trailing_zeros() as u8);
        m &= m - 1;
    }
    cards
}

#[inline]
pub fn bids_from_mask(mask: u16) -> Vec<u8> {
    let mut bids = Vec::with_capacity(mask.count_ones() as usize);
    let mut m = mask;
    while m != 0 {
        bids.push(m.trailing_zeros() as u8);
        m &= m - 1;
    }
    bids
}

#[inline]
pub fn table_bits(env: &FastEnv) -> u64 {
    let mut bits = 0u64;
    for i in 0..env.table_len as usize {
        bits |= 1u64 << env.table[i].1;
    }
    bits
}

#[inline]
pub fn closest_legal_bid(mask: u16, target: f32) -> u8 {
    let bids = bids_from_mask(mask);
    *bids
        .iter()
        .min_by(|&&a, &&b| {
            let da = (a as f32 - target).abs();
            let db = (b as f32 - target).abs();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap_or(&0)
}

/// Select an action (bid in Bidding phase, card_idx in Playing phase) for `level`.
pub fn bot_action(env: &FastEnv, level: u8) -> u8 {
    match env.phase {
        Phase::Bidding => match level {
            BOT_ROOKIE => rookie_bid(env),
            BOT_REGULAR => regular_bid(env),
            BOT_TACTICIAN | BOT_FINAL_BOSS => tactician_bid(env),
            _ => tactician_bid(env),
        },
        Phase::Playing => match level {
            BOT_ROOKIE => rookie_card(env),
            BOT_REGULAR => regular_card(env),
            BOT_TACTICIAN => tactician_card(env),
            BOT_FINAL_BOSS => final_boss_card(env),
            _ => tactician_card(env),
        },
        Phase::Terminal => 0,
    }
}

pub fn rookie_bid(env: &FastEnv) -> u8 {
    let p = env.current_player as usize;
    let mask = env.legal_bids_mask(p);
    let bids = bids_from_mask(mask);
    if bids.is_empty() {
        return 0;
    }
    let cap = (env.round_cards / 3).max(1);
    let mut rng = rand::thread_rng();
    let low_bids: Vec<u8> = bids.iter().copied().filter(|&b| b <= cap).collect();
    if !low_bids.is_empty() {
        *low_bids.choose(&mut rng).unwrap()
    } else {
        *bids.choose(&mut rng).unwrap()
    }
}

pub fn rookie_card(env: &FastEnv) -> u8 {
    let p = env.current_player as usize;
    let cards = cards_from_mask(env.legal_cards_mask(p));
    let mut rng = rand::thread_rng();
    *cards.choose(&mut rng).unwrap_or(&0)
}

pub fn regular_bid(env: &FastEnv) -> u8 {
    let p = env.current_player as usize;
    let mask = env.legal_bids_mask(p);
    if mask == 0 {
        return 0;
    }
    if env.round_cards == 1 {
        let mut max_opp_rank = 0u8;
        for i in 1..=3 {
            let opp = (p + i) % 4;
            let hand = env.hands[opp];
            if hand != 0 {
                max_opp_rank = max_opp_rank.max(rank_of(hand.trailing_zeros() as u8));
            }
        }
        let target = if max_opp_rank < 12 { 1.0 } else { 0.0 };
        return closest_legal_bid(mask, target);
    }

    let mut expected = 0.0f32;
    let mut hand = env.hands[p];
    while hand != 0 {
        let c = hand.trailing_zeros() as u8;
        hand &= hand - 1;
        expected += match rank_of(c) {
            14 => 0.85,
            13 => 0.55,
            12 => 0.25,
            _ => 0.0,
        };
    }
    closest_legal_bid(mask, expected)
}

pub fn regular_card(env: &FastEnv) -> u8 {
    let p = env.current_player as usize;
    let legal = env.legal_cards_mask(p);
    let cards = cards_from_mask(legal);
    if cards.len() <= 1 {
        return *cards.first().unwrap_or(&0);
    }
    let bid = env.bids[p].max(0) as i32;
    let tricks = env.tricks_won[p] as i32;

    if env.table_len == 0 {
        return if tricks < bid {
            *cards.iter().max_by_key(|&&c| rank_of(c)).unwrap()
        } else {
            *cards.iter().min_by_key(|&&c| rank_of(c)).unwrap()
        };
    }

    let led = env.led_suit;
    let best = (0..env.table_len as usize)
        .map(|i| env.table[i].1)
        .filter(|&c| suit_of(c) == led)
        .map(|c| c % 13)
        .max()
        .unwrap_or(0);

    if tricks < bid {
        let mut winners: Vec<u8> = cards
            .iter()
            .copied()
            .filter(|&c| suit_of(c) == led && (c % 13) > best)
            .collect();
        if !winners.is_empty() {
            winners.sort_by_key(|&c| c % 13);
            *winners.last().unwrap()
        } else {
            *cards.iter().min_by_key(|&&c| rank_of(c)).unwrap()
        }
    } else {
        let mut ducks: Vec<u8> = cards
            .iter()
            .copied()
            .filter(|&c| suit_of(c) != led || (c % 13) < best)
            .collect();
        if !ducks.is_empty() {
            ducks.sort_by_key(|&c| c % 13);
            *ducks.last().unwrap()
        } else {
            *cards.iter().min_by_key(|&&c| rank_of(c)).unwrap()
        }
    }
}

pub fn tactician_bid(env: &FastEnv) -> u8 {
    let p = env.current_player as usize;
    let mask = env.legal_bids_mask(p);
    if mask == 0 {
        return 0;
    }
    if env.round_cards == 1 {
        return search_best_bid_pannplump(env, p, 49);
    }

    let hand = env.hands[p];
    let mut expected = 0.0f32;
    for suit in 0..4u8 {
        let s_cards = hand & suit_bits(suit);
        let len = s_cards.count_ones();
        let mut m = s_cards;
        while m != 0 {
            let c = m.trailing_zeros() as u8;
            m &= m - 1;
            expected += match rank_of(c) {
                14 => 0.88,
                13 => {
                    if len >= 2 {
                        0.68
                    } else {
                        0.38
                    }
                }
                12 => {
                    if len >= 3 {
                        0.42
                    } else if len == 2 {
                        0.22
                    } else {
                        0.18
                    }
                }
                11 => {
                    if len >= 4 {
                        0.20
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
        }
        if env.round_cards >= 6 && len >= 4 {
            expected += 0.20 * (len - 3) as f32;
        }
    }

    // Context adjustment based on bids already placed.
    if env.bids_made > 0 {
        let sum_so_far: i32 = (0..4)
            .map(|q| env.bids[q])
            .filter(|&b| b >= 0)
            .map(|b| b as i32)
            .sum();
        let avg_expected_so_far =
            (env.round_cards as f32) * (env.bids_made as f32 / 4.0);
        if sum_so_far as f32 > avg_expected_so_far + 0.8 {
            expected -= 0.18;
        } else if (sum_so_far as f32) < avg_expected_so_far - 0.8 {
            expected += 0.12;
        }
    }

    closest_legal_bid(mask, expected.clamp(0.0, env.round_cards as f32))
}

#[inline]
pub fn is_boss_card(card: u8, unseen: u64) -> bool {
    let suit = suit_of(card);
    let higher_in_suit = suit_bits(suit) & !((1u64 << (card + 1)) - 1);
    (unseen & higher_in_suit) == 0
}

pub fn tactician_card(env: &FastEnv) -> u8 {
    let p = env.current_player as usize;
    let legal = env.legal_cards_mask(p);
    let cards = cards_from_mask(legal);
    if cards.len() <= 1 {
        return *cards.first().unwrap_or(&0);
    }

    let unseen = FULL_DECK & !env.hands[p] & !env.played & !table_bits(env);
    let bid = env.bids[p].max(0) as i32;
    let tricks = env.tricks_won[p] as i32;
    let need_tricks = tricks < bid;

    // 1. Leading the trick
    if env.table_len == 0 {
        if need_tricks {
            if let Some(&boss) = cards.iter().find(|&&c| is_boss_card(c, unseen)) {
                return boss;
            }
            return *cards.iter().max_by_key(|&&c| rank_of(c)).unwrap();
        } else {
            // Ducking lead: prefer a non-boss low card.
            if let Some(&safe_low) = cards
                .iter()
                .filter(|&&c| !is_boss_card(c, unseen))
                .min_by_key(|&&c| rank_of(c))
            {
                return safe_low;
            }
            return *cards.iter().min_by_key(|&&c| rank_of(c)).unwrap();
        }
    }

    // 2. Following in the trick
    let led = env.led_suit;
    let best = (0..env.table_len as usize)
        .map(|i| env.table[i].1)
        .filter(|&c| suit_of(c) == led)
        .map(|c| c % 13)
        .max()
        .unwrap_or(0);

    if need_tricks {
        let mut winners: Vec<u8> = cards
            .iter()
            .copied()
            .filter(|&c| suit_of(c) == led && (c % 13) > best)
            .collect();
        winners.sort_by_key(|&c| c % 13);

        if !winners.is_empty() {
            if env.table_len == 3 {
                // 4th seat finesse: win with the cheapest card!
                return winners[0];
            }
            if let Some(&boss) = winners.iter().find(|&&c| is_boss_card(c, unseen)) {
                return boss;
            }
            return *winners.last().unwrap();
        }
        // Cannot beat the trick: dump lowest non-boss card.
        return *cards
            .iter()
            .min_by_key(|&&c| (is_boss_card(c, unseen) as u8, rank_of(c)))
            .unwrap();
    } else {
        // Trying to duck!
        let off_suit: Vec<u8> = cards
            .iter()
            .copied()
            .filter(|&c| suit_of(c) != led)
            .collect();
        if !off_suit.is_empty() {
            // Void in led suit: slough our highest/most dangerous off-suit card!
            return *off_suit
                .iter()
                .max_by_key(|&&c| (is_boss_card(c, unseen) as u8, rank_of(c)))
                .unwrap();
        }

        let mut under: Vec<u8> = cards
            .iter()
            .copied()
            .filter(|&c| (c % 13) < best)
            .collect();
        if !under.is_empty() {
            under.sort_by_key(|&c| c % 13);
            return *under.last().unwrap();
        }

        // Forced to overtake `best` in the led suit:
        if env.table_len == 3 {
            // Last to play: guaranteed to take this unwanted trick, so dump highest!
            return *cards.iter().max_by_key(|&&c| rank_of(c)).unwrap();
        }
        // Otherwise play lowest above `best`, hoping the next player overtakes us.
        *cards.iter().min_by_key(|&&c| rank_of(c)).unwrap()
    }
}

pub fn final_boss_card(env: &FastEnv) -> u8 {
    let p = env.current_player as usize;
    let legal = env.legal_cards_mask(p);
    if legal.count_ones() <= 1 {
        return legal.trailing_zeros() as u8;
    }
    let remaining = env.round_cards.saturating_sub(env.current_trick);
    if remaining <= 5 {
        search_best_card(env, p, 12, None)
    } else {
        tactician_card(env)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::card_index;

    #[test]
    fn test_rookie_bot_action() {
        let mut env = FastEnv::default();
        let mut rng = rand::thread_rng();
        env.new_round(5, 0, &mut rng);
        let bid = bot_action(&env, BOT_ROOKIE);
        assert!((env.legal_bids_mask(0) & (1u16 << bid)) != 0);

        for _ in 0..4 {
            let p = env.current_player as usize;
            let b = rookie_bid(&env);
            assert!((env.legal_bids_mask(p) & (1u16 << b)) != 0);
            env.step_bid(b);
        }

        assert_eq!(env.phase, Phase::Playing);
        let p = env.current_player as usize;
        let card = bot_action(&env, BOT_ROOKIE);
        assert!((env.legal_cards_mask(p) & (1u64 << card)) != 0);
    }

    #[test]
    fn test_regular_forehead_bid() {
        let mut env = FastEnv::default();
        env.round_cards = 1;
        env.start_bidder = 0;
        env.current_player = 0;
        env.phase = Phase::Bidding;
        env.hands[0] = 1u64 << card_index(0, 10);
        // Opponents hold 2H, 3H, 4H (ranks < 12)
        env.hands[1] = 1u64 << card_index(0, 2);
        env.hands[2] = 1u64 << card_index(0, 3);
        env.hands[3] = 1u64 << card_index(0, 4);

        let bid = regular_bid(&env);
        assert_eq!(bid, 1);

        // If one opponent holds Ace (rank 14 >= 12)
        env.hands[1] = 1u64 << card_index(0, 14);
        let bid_ace = regular_bid(&env);
        assert_eq!(bid_ace, 0);
    }

    #[test]
    fn test_tactician_finesse_and_slough() {
        let mut env = FastEnv::default();
        env.round_cards = 2;
        env.bids = [1, 1, 0, 1];
        env.phase = Phase::Playing;
        env.current_player = 3;
        env.table_len = 3;
        env.led_suit = 0; // hearts
        env.table[0] = (0, card_index(0, 5));
        env.table[1] = (1, card_index(0, 9));
        env.table[2] = (2, card_index(0, 7));
        // Seat 3 needs a trick (bid 1, won 0).
        // Best rank so far is 9 (card_index(0, 9)).
        // Seat 3 holds 10 of hearts (rank 10) and Ace of hearts (rank 14).
        // In 4th seat, tactician finesses with cheapest winning card: 10 of hearts!
        env.hands[3] = (1u64 << card_index(0, 10)) | (1u64 << card_index(0, 14));
        let card = tactician_card(&env);
        assert_eq!(card, card_index(0, 10));

        // Now test sloughing: seat 3 has already made their bid (bid 0, won 0).
        // Seat 3 is void in hearts, holds Spades 2 and Spades Ace.
        // Trying to duck, void in led suit -> sloughs dangerous high card (Spades Ace).
        env.bids[3] = 0;
        env.hands[3] = (1u64 << card_index(1, 2)) | (1u64 << card_index(1, 14));
        let slough_card = tactician_card(&env);
        assert_eq!(slough_card, card_index(1, 14));
    }

    #[test]
    fn test_is_boss_card() {
        let unseen = (1u64 << card_index(0, 10)) | (1u64 << card_index(0, 5));
        // Ace (rank 14) is boss card: no higher unseen
        assert!(is_boss_card(card_index(0, 14), unseen));
        // Jack (rank 11) is boss card: no higher unseen
        assert!(is_boss_card(card_index(0, 11), unseen));
        // 9 is NOT boss card: 10 is unseen
        assert!(!is_boss_card(card_index(0, 9), unseen));
    }
}
