//! Rollout throughput benchmark (§7.1): a full round of `batch` games in
//! lockstep, measuring steps/s. Run with:
//!
//! `cargo run --release --example bench_rollout -- [batch] [players] [cards] [rounds]`

use plump_engine::encode::K_MAX;
use plump_engine::rollout::Rollout;
use plump_engine::scoring::ScoringConfig;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16_384);
    let n_players: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let n_cards: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let rounds: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);

    let mut r = Rollout::new(n_players, n_cards, batch, 42, ScoringConfig::default());
    let d = r.decisions_per_round();
    let mut out_idx = vec![0i16; batch * K_MAX];
    let mut out_len = vec![0u16; batch];
    let mut actions = vec![0u8; batch];

    // warm-up round
    for _ in 0..d {
        for (g, a) in actions.iter_mut().enumerate() {
            let mask = if r.is_bid(g) {
                r.legal_bid(g) as u64
            } else {
                r.legal_play(g)
            };
            *a = mask.trailing_zeros() as u8;
        }
        r.encode(&mut out_idx, &mut out_len);
        r.step(&actions);
    }
    r.reset(0);

    let t0 = Instant::now();
    let mut total_steps = 0usize;
    for rd in 0..rounds {
        for _ in 0..d {
            for (g, a) in actions.iter_mut().enumerate() {
                let mask = if r.is_bid(g) {
                    r.legal_bid(g) as u64
                } else {
                    r.legal_play(g)
                };
                *a = mask.trailing_zeros() as u8;
            }
            r.encode(&mut out_idx, &mut out_len);
            r.step(&actions);
            total_steps += 1;
        }
        assert!(r.round_over(), "round {rd} did not finish");
        if rd + 1 < rounds {
            r.reset(rd as u64 + 1);
        }
    }
    let dt = t0.elapsed();
    let steps_s = total_steps as f64 / dt.as_secs_f64();
    let games_s = batch as f64 * steps_s;
    println!(
        "played {rounds} rounds of {batch} games (P={n_players}, C={n_cards}): {total_steps} steps in {dt:?}"
    );
    println!("steps/s: {steps_s:.0}  |  game-decisions/s: {games_s:.0}");
}
