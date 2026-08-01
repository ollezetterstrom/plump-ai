//! Feature encoder (§3.6) and batch container.
//!
//! `encode_one` turns one `RoundState` (plus its `MatchContext`) into a
//! fixed-width `[i16; K_MAX]` index row and a length. Every index is emitted
//! at an **actor-relative** seat offset (§3.5, slot 0 is always "me") with
//! canonical card indices, and by **bit-iteration only** (§3.6). Trailing
//! slots are padded with `N_FEATURES` (the net's no-op `padding_idx`), so the
//! host→device copy is one contiguous `[B, K_MAX]` tensor with static shape.
//!
//! Sub-block schema (all documented offsets are relative to the block base in
//! the authoritative `LAYOUT` registry, §5.2):
//!
//! ```text
//! role_flags  (1035, 30 = 5×6 one-hots)
//!   1035+0..6   is_dealer[rel]            one-hot over rel seats
//!   1035+6..12  is_leader[rel]            one-hot over rel seats
//!   1035+12..18 actor position in trick   one-hot (0 = about to lead)
//!   1035+18..24 actor bid-order position  one-hot
//!   1035+24..30 n_players                 one-hot (3..=6)
//! round_ctx   (1065, 40)
//!   1065+0..6   n_players                 one-hot
//!   1065+6..17  n_cards                   one-hot (0..=10)
//!   1065+17..27 undealt_count bucket      one-hot (52-PC)/5
//!   1065+27..38 tricks_remaining          one-hot (0..=10)
//!   1065+38..40 phase                     one-hot (bidding, playing)
//! match_ctx   (1105, 120)                 simulated context (§8, Stage A)
//!   1105+0..16  own cumulative score      bucket own/16
//!   1105+16..76 opp rel score buckets     5 seats × 12, rel = opp - own
//!   1105+76..87 rounds_remaining          one-hot (0..=10)
//!   1105+87..98 round_size                one-hot (0..=10)
//!   1105+98..104 rank                     one-hot (0..=5)
//!   1105+104..120 schedule_sum            bucket /5
//! ```

use crate::cards::{bit, CardSet, FULL_DECK, MAX_SEATS, N_CARDS, N_SUITS, SUIT_MASK};
use crate::game::{Phase, RoundState};
use crate::layout::N_FEATURES;
use crate::prng::Rng;

/// Hard cap on active features per row, so the host→device copy is a single
/// contiguous `[B, K_MAX]` transfer with static shape.
pub const K_MAX: usize = 128;

/// Match-level context for one game (§8). Populated from a *simulated* random
/// match context during Stage A so the network learns to read the features.
#[derive(Clone, Copy, Debug)]
pub struct MatchContext {
    /// Cumulative score per absolute seat.
    pub scores: [i32; MAX_SEATS],
    /// Rounds remaining in the ladder (including the current one).
    pub rounds_remaining: u8,
    /// Card count of the current round.
    pub round_size: u8,
    /// Position in the standings, 0 = best.
    pub rank: u8,
    /// Sum of remaining round sizes (current round onwards).
    pub schedule_sum: u8,
}

impl Default for MatchContext {
    fn default() -> Self {
        Self {
            scores: [0; MAX_SEATS],
            rounds_remaining: 0,
            round_size: 0,
            rank: 0,
            schedule_sum: 0,
        }
    }
}

impl MatchContext {
    /// Plausible random ladder context for the given player count.
    pub fn simulated(p: u8, rng: &mut Rng) -> Self {
        let mut scores = [0i32; MAX_SEATS];
        for s in scores.iter_mut() {
            *s = rng.next_below(201) as i32; // cumulative score 0..=200
        }
        Self {
            scores,
            rounds_remaining: rng.next_below(11) as u8,
            round_size: 1 + rng.next_below(10) as u8,
            rank: rng.next_below(p as usize) as u8,
            schedule_sum: rng.next_below(80) as u8,
        }
    }
}

/// A batch of games to encode. Owned `RoundState`s for now; Phase 3 swaps the
/// storage for pinned, double-buffered SoA buffers behind the same interface.
pub struct Batch {
    pub games: Vec<RoundState>,
    pub match_ctx: Vec<MatchContext>,
}

impl Batch {
    pub fn new(games: Vec<RoundState>, match_ctx: Vec<MatchContext>) -> Self {
        assert_eq!(
            games.len(),
            match_ctx.len(),
            "games/match_ctx length mismatch"
        );
        Self { games, match_ctx }
    }

    /// Owned `Batch` from slices (clones; the `Rollout` uses slice encoding
    /// directly and avoids this).
    pub fn from_parts(games: &[RoundState], match_ctx: &[MatchContext]) -> Self {
        assert_eq!(
            games.len(),
            match_ctx.len(),
            "games/match_ctx length mismatch"
        );
        Self {
            games: games.to_vec(),
            match_ctx: match_ctx.to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }
}

/// Encodes one game into `out` (≥ `K_MAX` long), returns the active length and
/// pads the remainder with `N_FEATURES`.
pub fn encode_one(g: &RoundState, m: &MatchContext, out: &mut [i16]) -> usize {
    debug_assert!(out.len() >= K_MAX, "output row too small");
    let mut e = Enc {
        g,
        m,
        p: g.n_players as usize,
        actor: g.actor,
        out,
        len: 0,
    };
    e.emit();
    debug_assert!(e.len <= K_MAX, "K_MAX exceeded: {}", e.len);
    for slot in e.out.iter_mut().take(K_MAX).skip(e.len) {
        *slot = N_FEATURES as i16;
    }
    e.len
}

/// Encodes a whole batch in parallel (§3.6): pre-allocated `[B, K_MAX]` index
/// buffer plus one length per game, zero allocation per step.
pub fn encode_batch(batch: &Batch, out_idx: &mut [i16], out_len: &mut [u16]) {
    encode_batch_ref(&batch.games, &batch.match_ctx, out_idx, out_len);
}

/// Slice-level `encode_batch`, used by the `Batch` wrapper and by the lockstep
/// `Rollout` (which owns its games directly).
pub fn encode_batch_ref(
    games: &[RoundState],
    match_ctx: &[MatchContext],
    out_idx: &mut [i16],
    out_len: &mut [u16],
) {
    assert_eq!(
        games.len(),
        match_ctx.len(),
        "games/match_ctx length mismatch"
    );
    assert_eq!(out_idx.len(), games.len() * K_MAX, "out_idx wrong size");
    assert_eq!(out_len.len(), games.len(), "out_len wrong size");
    use rayon::prelude::*;
    out_idx
        .par_chunks_mut(K_MAX)
        .zip(out_len.par_iter_mut())
        .enumerate()
        .for_each(|(g, (slot, len))| {
            *len = encode_one(&games[g], &match_ctx[g], slot) as u16;
        });
}

/// Per-row encoder state. `rel` maps an absolute seat to its actor-relative
/// offset so slot 0 is always "me" (§3.5).
struct Enc<'a> {
    g: &'a RoundState,
    m: &'a MatchContext,
    p: usize,
    actor: u8,
    out: &'a mut [i16],
    len: usize,
}

impl Enc<'_> {
    #[inline]
    fn rel(&self, abs: u8) -> usize {
        ((abs + self.p as u8 - self.actor) % self.p as u8) as usize
    }

    #[inline]
    fn push(&mut self, idx: usize) {
        debug_assert!(idx < N_FEATURES, "index out of range");
        self.out[self.len] = idx as i16;
        self.len += 1;
    }

    fn emit(&mut self) {
        self.emit_my_hand();
        self.emit_played_by_rel();
        self.emit_trick_cards();
        self.emit_led_suit();
        self.emit_my_voids();
        self.emit_other_voids();
        self.emit_unseen_by_suit();
        self.emit_bids_rel();
        self.emit_tricks_rel();
        self.emit_need_rel();
        self.emit_role_flags();
        self.emit_round_ctx();
        self.emit_match_ctx();
    }

    /// Bit-iteration over a card set (§3.6): one push per set bit.
    #[inline]
    fn push_cards(&mut self, base: usize, outer: usize, set: CardSet) {
        let mut m = set;
        while m != 0 {
            let c = m.trailing_zeros() as usize;
            m &= m - 1;
            self.push(base + outer * N_CARDS + c);
        }
    }

    fn emit_my_hand(&mut self) {
        self.push_cards(0, 0, self.g.hands[self.actor as usize]);
    }

    fn emit_played_by_rel(&mut self) {
        for abs in 0..self.p as u8 {
            let r = self.rel(abs);
            self.push_cards(52, r, self.g.public.played_by[abs as usize]);
        }
    }

    /// Current trick, indexed by play position (leader = 0), so seat
    /// relabeling is a no-op here by construction.
    fn emit_trick_cards(&mut self) {
        for pos in 0..self.g.trick_count as usize {
            self.push_cards(364, pos, bit(self.g.trick_cards[pos]));
        }
    }

    fn emit_led_suit(&mut self) {
        let slot = match self.g.led {
            Some(s) => s as usize,
            None => 4, // "none"
        };
        self.push(676 + slot);
    }

    fn emit_my_voids(&mut self) {
        let hand = self.g.hands[self.actor as usize];
        for (s, mask) in SUIT_MASK.iter().enumerate() {
            if hand & mask == 0 {
                self.push(681 + s);
            }
        }
    }

    fn emit_other_voids(&mut self) {
        for abs in 0..self.p as u8 {
            if abs == self.actor {
                continue;
            }
            let r = self.rel(abs);
            let v = self.g.public.voids[abs as usize];
            for (s, mask) in SUIT_MASK.iter().enumerate() {
                if v & mask != 0 {
                    self.push(685 + r * N_SUITS + s);
                }
            }
        }
    }

    /// Per suit, the one-hot count of cards not in my hand and not played
    /// (0..13) — opponents' hands + dead stock (§5.3).
    fn emit_unseen_by_suit(&mut self) {
        let unseen = FULL_DECK & !self.g.hands[self.actor as usize] & !self.g.public.all_played;
        for (s, mask) in SUIT_MASK.iter().enumerate() {
            let cnt = (unseen & mask).count_ones() as usize;
            self.push(709 + s * 14 + cnt);
        }
    }

    fn emit_bids_rel(&mut self) {
        for abs in 0..self.p as u8 {
            let r = self.rel(abs);
            let base = 765 + r * 13;
            if self.g.has_bid[abs as usize] {
                self.push(base + self.g.bids[abs as usize] as usize);
            } else {
                self.push(base + 11); // "unknown" (bin 12 reserved)
            }
        }
    }

    fn emit_tricks_rel(&mut self) {
        for abs in 0..self.p as u8 {
            let r = self.rel(abs);
            self.push(843 + r * 11 + self.g.tricks_won[abs as usize] as usize);
        }
    }

    fn emit_need_rel(&mut self) {
        for abs in 0..self.p as u8 {
            if !self.g.has_bid[abs as usize] {
                continue;
            }
            let r = self.rel(abs);
            let need = (i32::from(self.g.bids[abs as usize])
                - i32::from(self.g.tricks_won[abs as usize]))
            .clamp(-10, 10);
            self.push(909 + r * 21 + (need + 10) as usize);
        }
    }

    fn emit_role_flags(&mut self) {
        // is_dealer[rel]
        self.push(1035 + self.rel(self.g.dealer));
        // is_leader[rel]
        let leader = if self.g.trick_count > 0 {
            self.g.trick_seats[0]
        } else {
            self.g.trick_leader
        };
        self.push(1035 + 6 + self.rel(leader));
        // actor position in the current trick (next-to-play index).
        self.push(1035 + 12 + self.g.trick_count as usize);
        // actor bid-order position: bidding starts at (dealer + 1) mod P.
        let bid_order = ((i32::from(self.actor) - (i32::from(self.g.dealer) + 1)) % self.p as i32
            + self.p as i32)
            % self.p as i32;
        self.push(1035 + 18 + bid_order as usize);
        // n_players one-hot.
        self.push(1035 + 24 + self.p.min(5));
    }

    fn emit_round_ctx(&mut self) {
        // n_players one-hot.
        self.push(1065 + self.p.min(5));
        // n_cards one-hot.
        self.push(1065 + 6 + self.g.n_cards as usize);
        // undealt_count = 52 - P*C, bucketed.
        let bucket = (self.g.public.undealt_count as usize / 5).min(9);
        self.push(1065 + 6 + 11 + bucket);
        // tricks_remaining.
        let remaining = self.g.n_cards as usize - self.g.tricks_played as usize;
        self.push(1065 + 6 + 11 + 10 + remaining);
        // phase.
        match self.g.phase {
            Phase::Bidding => self.push(1065 + 38),
            Phase::Playing => self.push(1065 + 38 + 1),
            Phase::Done => {} // never encoded: no decision is pending
        }
    }

    fn emit_match_ctx(&mut self) {
        let own = self.m.scores[self.actor as usize];
        // own cumulative score bucket.
        self.push(1105 + (own.clamp(0, 240) / 16) as usize);
        // opp rel score buckets (actor-relative seats).
        for abs in 0..self.p as u8 {
            if abs == self.actor {
                continue;
            }
            let r = self.rel(abs);
            let rel = (self.m.scores[abs as usize] - own).clamp(-60, 60);
            self.push(1105 + 16 + r * 12 + ((rel + 60) / 10) as usize);
        }
        // rounds_remaining one-hot.
        self.push(1105 + 76 + (self.m.rounds_remaining as usize).min(10));
        // round_size one-hot.
        self.push(1105 + 87 + (self.m.round_size as usize).min(10));
        // rank one-hot.
        self.push(1105 + 98 + (self.m.rank as usize).min(5));
        // schedule_sum bucket.
        self.push(1105 + 104 + (self.m.schedule_sum as usize / 5).min(15));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Phase;
    use crate::layout::{all_permutations_of_4, build_perm_table, LAYOUT};
    use crate::scoring::ScoringConfig;

    fn random_game(seed: u64, n_players: u8, n_cards: u8, stop: usize) -> RoundState {
        let mut rng = Rng::new(seed);
        let dealer = (seed % n_players as u64) as u8;
        let mut g = RoundState::new(n_players, n_cards, dealer, seed, ScoringConfig::default());
        for _ in 0..stop {
            if g.is_done() {
                break;
            }
            let mask = match g.phase {
                Phase::Bidding => g.legal_bid_mask(g.actor) as u64,
                Phase::Playing => g.legal_play_mask(g.actor),
                Phase::Done => unreachable!(),
            };
            let mut bits = Vec::new();
            let mut m = mask;
            while m != 0 {
                let b = m.trailing_zeros();
                bits.push(b as u8);
                m &= m - 1;
            }
            let act = bits[rng.next_below(bits.len())];
            g.step(act);
        }
        g
    }

    fn encode_raw(g: &RoundState, m: &MatchContext) -> Vec<i16> {
        let mut row = [0i16; K_MAX];
        let len = encode_one(g, m, &mut row);
        row[..len].to_vec()
    }

    fn encode_sorted(g: &RoundState) -> Vec<i16> {
        let mut v = encode_raw(g, &MatchContext::default());
        v.sort_unstable();
        v
    }

    #[test]
    fn every_emitted_index_claimed_by_exactly_one_block() {
        for seed in 0..300u64 {
            let total = 5 * (1 + 8);
            let stop = (seed as usize * 7) % (total - 1);
            let g = random_game(seed, 5, 8, stop);
            for idx in encode_raw(&g, &MatchContext::default()) {
                assert!(idx < N_FEATURES as i16, "index escaped layout");
                let mut claimed = 0;
                for b in LAYOUT {
                    let i = idx as usize;
                    if i >= b.offset && i < b.offset + b.size {
                        claimed += 1;
                    }
                }
                assert_eq!(claimed, 1, "index {idx} not claimed by exactly one block");
            }
        }
    }

    #[test]
    fn k_max_never_exceeded_worst_case() {
        // P=6,C=8 and P=5,C=10 are the largest legal configurations.
        for (p, c) in [(6u8, 8u8), (5u8, 10u8)] {
            let total = p as usize * (1 + c as usize);
            for seed in 0..400u64 {
                let stop = (seed as usize * 13) % (total - 1);
                let g = random_game(seed, p, c, stop);
                let row = encode_raw(&g, &MatchContext::default());
                assert!(
                    row.len() <= K_MAX,
                    "K_MAX exceeded ({}) at seed {seed} p={p} c={c}",
                    row.len()
                );
            }
        }
    }

    #[test]
    fn seat_relabeling_is_invariant() {
        for seed in 0..100u64 {
            let total = 5 * (1 + 9);
            let stop = (seed as usize * 11) % (total - 1);
            let g = random_game(seed, 5, 9, stop);
            let base = encode_sorted(&g);
            for k in 1..5u8 {
                let mut h = g.clone();
                h.rotate_seats(k);
                assert_eq!(encode_sorted(&h), base, "relabel k={k} seed={seed}");
            }
        }
    }

    #[test]
    fn suit_permutation_parity() {
        let t = build_perm_table();
        for seed in 0..60u64 {
            let total = 5 * (1 + 7);
            let stop = (seed as usize * 17) % (total - 1);
            let g = random_game(seed, 5, 7, stop);
            for (pi, perm) in all_permutations_of_4().into_iter().enumerate() {
                let mut h = g.clone();
                h.relabel_suits(&perm);
                let relabeled = encode_sorted(&h);
                let mut permuted: Vec<i16> = encode_raw(&g, &MatchContext::default())
                    .into_iter()
                    .map(|i| t[pi * N_FEATURES + i as usize])
                    .collect();
                permuted.sort_unstable();
                assert_eq!(
                    relabeled, permuted,
                    "parity failed seed={seed} perm={perm:?}"
                );
            }
        }
    }

    #[test]
    fn batch_matches_single_and_pads() {
        let mut games = Vec::new();
        let mut ctx = Vec::new();
        for seed in 0..64u64 {
            games.push(random_game(seed, 4, 6, seed as usize % (4 * 7 - 1)));
            ctx.push(MatchContext::simulated(4, &mut Rng::new(seed)));
        }
        let batch = Batch::new(games.clone(), ctx.clone());
        let mut idx = vec![0i16; 64 * K_MAX];
        let mut len = vec![0u16; 64];
        encode_batch(&batch, &mut idx, &mut len);
        for g in 0..64 {
            assert_eq!(len[g] as usize, encode_raw(&games[g], &ctx[g]).len());
            let mut row = [0i16; K_MAX];
            encode_one(&games[g], &ctx[g], &mut row);
            assert_eq!(&idx[g * K_MAX..(g + 1) * K_MAX], &row[..]);
            // trailing slots are the padding index.
            let n = len[g] as usize;
            assert!(idx[g * K_MAX + n..(g + 1) * K_MAX]
                .iter()
                .all(|&v| v == N_FEATURES as i16));
        }
    }

    #[test]
    fn match_ctx_extremes_stay_in_bounds() {
        let g = random_game(1, 3, 5, 3);
        let m = MatchContext {
            scores: [i32::MAX / 2; MAX_SEATS],
            rounds_remaining: 255,
            round_size: 255,
            rank: 255,
            schedule_sum: 255,
        };
        let row = encode_raw(&g, &m);
        for idx in row.iter().copied() {
            let i = idx as usize;
            // Everything from the match_ctx base on must stay inside the block.
            if i >= 1105 {
                assert!(i < 1105 + 120, "match_ctx index out of bounds: {idx}");
            }
        }
        // The extreme context must hit the top bucket of every match_ctx group
        // (opponents score identically, so rel = 0 -> bucket 6 per rel seat).
        for expected in [
            1105 + 15,              // own cumulative bucket
            1105 + 16 + 12 + 6,     // opp rel seat 1
            1105 + 16 + 2 * 12 + 6, // opp rel seat 2
            1105 + 76 + 10,         // rounds_remaining
            1105 + 87 + 10,         // round_size
            1105 + 98 + 5,          // rank
            1105 + 104 + 15,        // schedule_sum
        ] {
            assert!(
                row.contains(&(expected as i16)),
                "missing match_ctx slot {expected}"
            );
        }
    }
}
