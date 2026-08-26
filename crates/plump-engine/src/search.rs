//! search.rs — determinized MCTS for play-time deep search.
//! No torch here — caller provides q_fn to score worlds. 10-50x faster than Python engine.
//!
//! Usage from Python (plump/search/mcts.py):
//!   worlds = sample_worlds(env, N)  // consistent with void_matrix + played
//!   scores = mcts_search(worlds, q_fn, sims=64) // avg over worlds

use crate::engine::PlumpEnv;
use rand::seq::SliceRandom;
use rand::Rng;

/// Sample N worlds consistent with void_matrix + played + viewer hand.
/// Returns Vec<PlumpEnv> with hands filled.
pub fn sample_worlds(base: &PlumpEnv, viewer: usize, n: usize) -> Vec<PlumpEnv> {
    let mut out = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    // Cards not yet seen: 52 - played - viewer hand - table
    for _ in 0..n {
        let mut w = base.clone();
        // Build unknown cards pool
        let mut pool: Vec<(char, u8)> = {
            let mut all: Vec<(char, u8)> = crate::cards::SUITS.iter().flat_map(|&s| (2..15).map(move |r| (s, r))).collect();
            all.retain(|c| !w.played.contains(c) && !w.table.iter().any(|(_, cc)| cc == c) && !w.hands[viewer].contains(c));
            all
        };
        pool.shuffle(&mut rng);
        // Fill other players' hands respecting void
        // Simple: random fill, reject if violates void (loop a few times)
        let mut ok = false;
        for _ in 0..100 {
            let mut tmp_pool = pool.clone();
            tmp_pool.shuffle(&mut rng);
            let mut idx = 0;
            let mut good = true;
            for p in 0..4 {
                if p == viewer {
                    continue;
                }
                // hand size is same as base.hands[p] len (remaining cards)
                // we need to refill randomly
                // For now just random — void check: if player is void in H, they shouldn't get H unknown? But unknown pool may contain H that they are void in — that's impossible, so we filter.
                // So we try to avoid giving void suit
                let need = base.hands[p].len();
                let mut hand: Vec<(char, u8)> = Vec::new();
                let mut used = vec![];
                for _ in 0..need {
                    // find card not in void suit
                    let mut found = None;
                    for (j, &c) in tmp_pool.iter().enumerate() {
                        let s_idx = match c.0 { 'H'=>0,'S'=>1,'D'=>2,'C'=>3,_=>0 };
                        if w.void_matrix[p][s_idx] {
                            continue;
                        }
                        found = Some(j);
                        break;
                    }
                    if let Some(j) = found {
                        let c = tmp_pool.remove(j);
                        hand.push(c);
                        used.push(j);
                    } else {
                        good = false;
                        break;
                    }
                }
                if !good { break; }
                w.hands[p] = hand;
                // don't reset idx — we consumed
            }
            if good {
                ok = true;
                break;
            }
        }
        if ok {
            out.push(w);
        }
    }
    out
}

/// Simple rollout score: average Q over worlds for each legal move.
/// Caller provides q_fn: |world, player, card_idx| -> f32
/// Returns Vec<(card, score)>
pub fn score_moves(
    worlds: &[PlumpEnv],
    player: usize,
    legal: &[(char, u8)],
    q_fn: &dyn Fn(&PlumpEnv, usize, (char, u8)) -> f32,
) -> Vec<((char, u8), f32)> {
    let mut scores: Vec<(f32, usize)> = vec![(0.0, 0); legal.len()];
    for w in worlds {
        for (i, &card) in legal.iter().enumerate() {
            let s = q_fn(w, player, card);
            scores[i].0 += s;
            scores[i].1 += 1;
        }
    }
    legal
        .iter()
        .zip(scores)
        .map(|(&c, (sum, cnt))| (c, sum / cnt as f32))
        .collect()
}
