//! Throughput benchmark for the batch encoder (§3.6).
//!
//! Run: `cargo run --release --example bench_encode -- [states] [players] [cards]`
//!
//! Encodes `states` games (advanced to a mid-round position) repeatedly and
//! reports states/s. The plan's 2M states/s target assumes 16 cores; report on
//! this machine's core count.

use plump_engine::encode::{encode_batch, Batch, K_MAX};
use plump_engine::game::{Phase, RoundState};
use plump_engine::scoring::ScoringConfig;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let states: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16_384);
    let n_players: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let n_cards: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let total = n_players as usize * (1 + n_cards as usize);

    let mut games = Vec::with_capacity(states);
    for i in 0..states {
        let dealer = (i as u8) % n_players;
        let mut g = RoundState::new(
            n_players,
            n_cards,
            dealer,
            i as u64,
            ScoringConfig::default(),
        );
        let stop = (i % (total - 1)) + 1;
        for _ in 0..stop {
            if g.is_done() {
                break;
            }
            let mask = match g.phase {
                Phase::Bidding => g.legal_bid_mask(g.actor) as u64,
                Phase::Playing => g.legal_play_mask(g.actor),
                Phase::Done => unreachable!(),
            };
            g.step(mask.trailing_zeros() as u8);
        }
        games.push(g);
    }
    let mctx = vec![Default::default(); states];
    let batch = Batch::new(games, mctx);

    let mut out_idx = vec![0i16; states * K_MAX];
    let mut out_len = vec![0u16; states];
    encode_batch(&batch, &mut out_idx, &mut out_len); // warm-up

    let reps = 20;
    let t0 = Instant::now();
    for _ in 0..reps {
        encode_batch(&batch, &mut out_idx, &mut out_len);
    }
    let dt = t0.elapsed();
    let per_sec = states as f64 * reps as f64 / dt.as_secs_f64();
    println!("encoded {states} states x {reps} reps in {dt:?}");
    println!(
        "throughput: {per_sec:.0} states/s (threads = {})",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
}
