//! Bulk rollout driver (§7): plays `batch` games in lockstep and produces the
//! per-step tensors the trainer consumes.
//!
//! At every *decision point* the driver exposes, per game: the encoded
//! observation (into a caller-provided buffer, so Python can pin it), the
//! legal masks, the actor's absolute seat, and whether the next decision is a
//! bid. `step(actions)` advances each unfinished game by one decision and
//! recomputes the next snapshot. When a game finishes, its round scores are
//! recorded for the per-seat return (§4.1).
//!
//! All games share `(P, C)` and the same decision count `D = P·(P... )`,
//! actually `D = P * (1 + C)` (§3.2), so the batch is fully lockstep.

use crate::cards::{CardSet, MAX_SEATS};
use crate::encode::{encode_batch_ref, MatchContext};
use crate::game::{Phase, RoundState};
use crate::prng::Rng;
use crate::scoring::ScoringConfig;

/// Lockstep batch driver.
pub struct Rollout {
    pub n_players: u8,
    pub n_cards: u8,
    pub batch: usize,
    pub scoring: ScoringConfig,
    games: Vec<RoundState>,
    match_ctx: Vec<MatchContext>,
    /// Step index within the current round (0..D).
    pub step: usize,
    /// Games finished in the current round.
    pub done_count: usize,
    /// Round scores for finished games, `[batch * MAX_SEATS]`, zero until the
    /// round ends.
    scores: Vec<i32>,
    /// Number of rounds dealt so far.
    pub rounds_dealt: usize,
}

impl Rollout {
    /// Deals `batch` fresh games and computes the first decision snapshot.
    pub fn new(
        n_players: u8,
        n_cards: u8,
        batch: usize,
        seed: u64,
        scoring: ScoringConfig,
    ) -> Self {
        assert!(
            n_players >= 3 && n_players as usize <= MAX_SEATS,
            "bad n_players"
        );
        assert!(n_players as u16 * n_cards as u16 <= 52, "deck too small");
        let mut rng = Rng::new(seed);
        let mut games = Vec::with_capacity(batch);
        let mut match_ctx = Vec::with_capacity(batch);
        for g in 0..batch {
            // Per-game deal seed and simulated match context (§8, Stage A).
            let dealer = (g as u64 % n_players as u64) as u8;
            games.push(RoundState::new(
                n_players,
                n_cards,
                dealer,
                seed.wrapping_add(g as u64),
                scoring,
            ));
            match_ctx.push(MatchContext::simulated(n_players, &mut rng));
        }
        Self {
            n_players,
            n_cards,
            batch,
            scoring,
            games,
            match_ctx,
            step: 0,
            done_count: 0,
            scores: vec![0; batch * MAX_SEATS],
            rounds_dealt: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.batch
    }

    pub fn is_empty(&self) -> bool {
        self.batch == 0
    }

    pub fn game(&self, g: usize) -> &RoundState {
        &self.games[g]
    }

    /// Encodes the current snapshot into the caller-provided buffers
    /// (`out_idx.len() == batch * K_MAX`, `out_len.len() == batch`). Rows of
    /// finished games are fully padded. Returns the active length sum.
    pub fn encode(&self, out_idx: &mut [i16], out_len: &mut [u16]) {
        encode_batch_ref(&self.games, &self.match_ctx, out_idx, out_len);
    }

    /// A `Batch` view over the current games (for one-off encoding).
    pub fn as_batch(&self) -> crate::encode::Batch {
        crate::encode::Batch::from_parts(&self.games, &self.match_ctx)
    }

    /// Advances each unfinished game by one decision and recomputes the
    /// snapshot. `actions[g]` must be the next decision of `games[g]`.
    pub fn step(&mut self, actions: &[u8]) {
        assert_eq!(actions.len(), self.batch, "actions length mismatch");
        for (g, &action) in actions.iter().enumerate() {
            if self.games[g].is_done() {
                continue;
            }
            self.games[g].step(action);
            if self.games[g].is_done() {
                let sc = self.games[g].round_scores();
                let base = g * MAX_SEATS;
                for (p, s) in sc.iter().enumerate().take(self.n_players as usize) {
                    self.scores[base + p] = *s;
                }
                self.done_count += 1;
            }
        }
        self.step += 1;
    }

    /// True when every game has finished the current round.
    pub fn round_over(&self) -> bool {
        self.done_count == self.batch
    }

    /// Round scores for game `g` (valid once `round_over()`).
    pub fn round_scores(&self, g: usize) -> &[i32] {
        &self.scores[g * MAX_SEATS..g * MAX_SEATS + self.n_players as usize]
    }

    /// Re-deals every game for a new round; resets scores and the step
    /// counter. Returns the number of rounds dealt.
    pub fn reset(&mut self, seed: u64) -> usize {
        let mut rng = Rng::new(seed);
        for g in 0..self.batch {
            let dealer = (g as u64 % self.n_players as u64) as u8;
            self.games[g] = RoundState::new(
                self.n_players,
                self.n_cards,
                dealer,
                seed.wrapping_add(g as u64),
                self.scoring,
            );
            self.match_ctx[g] = MatchContext::simulated(self.n_players, &mut rng);
        }
        self.step = 0;
        self.done_count = 0;
        self.scores = vec![0; self.batch * MAX_SEATS];
        self.rounds_dealt += 1;
        self.rounds_dealt
    }

    /// Legal bid mask for game `g` at its current decision point (0 in play).
    pub fn legal_bid(&self, g: usize) -> u16 {
        if self.games[g].phase == Phase::Bidding {
            self.games[g].legal_bid_mask(self.games[g].actor)
        } else {
            0
        }
    }

    /// Legal play mask for game `g` at its current decision point (0 in bid).
    pub fn legal_play(&self, g: usize) -> CardSet {
        if self.games[g].phase == Phase::Playing {
            self.games[g].legal_play_mask(self.games[g].actor)
        } else {
            0
        }
    }

    /// Absolute seat to act next in game `g`.
    pub fn actor(&self, g: usize) -> u8 {
        self.games[g].actor
    }

    /// 1 if game `g`'s next decision is a bid.
    pub fn is_bid(&self, g: usize) -> bool {
        self.games[g].phase == Phase::Bidding
    }

    /// 1 if game `g` has finished the round.
    pub fn is_done(&self, g: usize) -> bool {
        self.games[g].is_done()
    }

    /// Total decisions per round for this configuration.
    pub fn decisions_per_round(&self) -> usize {
        self.n_players as usize * (1 + self.n_cards as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::K_MAX;

    #[test]
    fn full_round_lockstep_and_scores() {
        let n_players = 5u8;
        let n_cards = 3u8;
        let batch = 128usize;
        let mut r = Rollout::new(n_players, n_cards, batch, 7, ScoringConfig::default());
        let d = r.decisions_per_round();
        let mut out_idx = vec![0i16; batch * K_MAX];
        let mut out_len = vec![0u16; batch];
        for t in 0..d {
            assert_eq!(r.step, t);
            r.encode(&mut out_idx, &mut out_len);
            assert!(out_len[0] as usize <= K_MAX, "K_MAX exceeded");
            let mut actions = vec![0u8; batch];
            for (g, a) in actions.iter_mut().enumerate() {
                let mask = if r.is_bid(g) {
                    r.legal_bid(g) as u64
                } else {
                    r.legal_play(g)
                };
                assert_ne!(mask, 0, "empty mask at t={t} g={g}");
                *a = mask.trailing_zeros() as u8;
            }
            r.step(&actions);
        }
        assert!(r.round_over());
        for g in 0..batch {
            let sc = r.round_scores(g);
            for (p, s) in sc.iter().enumerate() {
                let game = r.game(g);
                assert_eq!(*s, game.scoring.score(game.bids[p], game.tricks_won[p]));
            }
        }
    }

    #[test]
    fn reset_deals_fresh_round() {
        let mut r = Rollout::new(4, 5, 64, 1, ScoringConfig::default());
        let before: Vec<CardSet> = (0..64).map(|g| r.game(g).hands[0]).collect();
        r.reset(2);
        let after: Vec<CardSet> = (0..64).map(|g| r.game(g).hands[0]).collect();
        assert!(before != after, "reset should re-deal");
        assert_eq!(r.step, 0);
        assert_eq!(r.rounds_dealt, 2);
    }

    #[test]
    fn done_rows_encode_safely_within_k_max() {
        // All games finish on the same final step, so a finished row is never
        // part of a live snapshot — but encoding one must stay in bounds.
        let mut r = Rollout::new(3, 2, 16, 3, ScoringConfig::default());
        let d = r.decisions_per_round();
        for _ in 0..d {
            let mut actions = vec![0u8; 16];
            for (g, a) in actions.iter_mut().enumerate() {
                let mask = if r.is_bid(g) {
                    r.legal_bid(g) as u64
                } else {
                    r.legal_play(g)
                };
                *a = mask.trailing_zeros() as u8;
            }
            r.step(&actions);
        }
        assert!(r.round_over());
        let mut out_idx = vec![0i16; 16 * K_MAX];
        let mut out_len = vec![0u16; 16];
        r.encode(&mut out_idx, &mut out_len);
        for g in 0..16 {
            let n = out_len[g] as usize;
            assert!(n <= K_MAX, "done row exceeds K_MAX");
            for &v in &out_idx[g * K_MAX..g * K_MAX + n] {
                assert!(v < crate::layout::N_FEATURES as i16, "index out of bounds");
            }
        }
    }
}
