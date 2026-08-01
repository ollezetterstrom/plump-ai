//! Brute-force correctness reference (§9.3): perfect-information enumeration
//! of the play tree for tiny configs (`P=3, C=2`). Every acting player chooses
//! the action maximizing their own score assuming all later players do the
//! same (subgame-perfect, no collusion) — the standard double-dummy value.
//! This is the absolute ground truth the trained agent is later checked
//! against; the Phase 1 exit criterion is that the tree is enumerable at all.

use crate::cards::{bit, suit, Card, CardSet, MAX_SEATS};
use crate::legal::legal_plays;
use crate::scoring::ScoringConfig;

/// Per-seat outcome of a perfect-information play-out.
#[derive(Clone, Copy, Debug)]
pub struct DdResult {
    pub scores: [i32; MAX_SEATS],
    pub tricks: [u8; MAX_SEATS],
}

/// Perfect-information (double-dummy) play-out of a full round.
///
/// `bids` must already be fixed (bidding is a shallow layer on top of this).
/// `hands` may be larger than `p`; only the first `p` seats participate.
pub fn dd_play(
    hands: &[CardSet; MAX_SEATS],
    bids: &[u8; MAX_SEATS],
    scoring: ScoringConfig,
    p: usize,
    c: usize,
) -> DdResult {
    DdState {
        hands: *hands,
        bids: *bids,
        tricks: [0; MAX_SEATS],
        leader: 0,
        trick: [0; MAX_SEATS],
        trick_seats: [0; MAX_SEATS],
        trick_count: 0,
        led: None,
        tricks_played: 0,
        p,
        c,
        scoring,
    }
    .minimax()
}

/// Convenience: just the per-seat scores.
pub fn dd_scores(
    hands: &[CardSet; MAX_SEATS],
    bids: &[u8; MAX_SEATS],
    scoring: ScoringConfig,
    p: usize,
    c: usize,
) -> [i32; MAX_SEATS] {
    dd_play(hands, bids, scoring, p, c).scores
}

/// Number of distinct legal play sequences for the given deal (leaf count of
/// the play tree). Used to prove the toy config is brute-force enumerable.
pub fn count_playout_leaves(hands: &[CardSet; MAX_SEATS], p: usize, c: usize) -> usize {
    DdState {
        hands: *hands,
        bids: [0; MAX_SEATS],
        tricks: [0; MAX_SEATS],
        leader: 0,
        trick: [0; MAX_SEATS],
        trick_seats: [0; MAX_SEATS],
        trick_count: 0,
        led: None,
        tricks_played: 0,
        p,
        c,
        scoring: ScoringConfig::default(),
    }
    .count()
}

#[derive(Clone)]
struct DdState {
    hands: [CardSet; MAX_SEATS],
    bids: [u8; MAX_SEATS],
    tricks: [u8; MAX_SEATS],
    leader: u8,
    trick: [Card; MAX_SEATS],
    trick_seats: [u8; MAX_SEATS],
    trick_count: u8,
    led: Option<u8>,
    tricks_played: u8,
    p: usize,
    c: usize,
    scoring: ScoringConfig,
}

impl DdState {
    /// Seat to act now: the leader opens each trick, then play moves clockwise.
    fn acting(&self) -> u8 {
        (self.leader + self.trick_count) % self.p as u8
    }

    fn play(&mut self, seat: u8, card: Card) {
        self.hands[seat as usize] &= !bit(card);
        self.trick[self.trick_count as usize] = card;
        self.trick_seats[self.trick_count as usize] = seat;
        self.trick_count += 1;
        if self.trick_count == 1 {
            self.led = Some(suit(card));
        }
        if self.trick_count as usize == self.p {
            self.resolve_trick();
        }
    }

    fn resolve_trick(&mut self) {
        let led_suit = suit(self.trick[0]);
        let mut winner = self.trick_seats[0];
        let mut best = self.trick[0];
        for i in 1..self.p {
            let c = self.trick[i];
            if suit(c) == led_suit && c > best {
                best = c;
                winner = self.trick_seats[i];
            }
        }
        self.tricks[winner as usize] += 1;
        self.tricks_played += 1;
        self.trick_count = 0;
        self.led = None;
        self.leader = winner;
    }

    fn minimax(&self) -> DdResult {
        if self.tricks_played as usize == self.c {
            let mut scores = [0i32; MAX_SEATS];
            for (p, slot) in scores.iter_mut().enumerate().take(self.p) {
                *slot = self.scoring.score(self.bids[p], self.tricks[p]);
            }
            return DdResult {
                scores,
                tricks: self.tricks,
            };
        }
        let seat = self.acting();
        let legal = legal_plays(self.hands[seat as usize], self.led);
        let mut best: Option<DdResult> = None;
        let mut bits = legal;
        while bits != 0 {
            let c = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let mut next = self.clone();
            next.play(seat, c);
            let res = next.minimax();
            if best.is_none_or(|b| res.scores[seat as usize] > b.scores[seat as usize]) {
                best = Some(res);
            }
        }
        best.expect("empty legal mask in double-dummy search")
    }

    fn count(&self) -> usize {
        if self.tricks_played as usize == self.c {
            return 1;
        }
        let seat = self.acting();
        let legal = legal_plays(self.hands[seat as usize], self.led);
        let mut n = 0;
        let mut bits = legal;
        while bits != 0 {
            let c = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let mut next = self.clone();
            next.play(seat, c);
            n += next.count();
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deal_3x2() -> [CardSet; MAX_SEATS] {
        let mut h = [0u64; MAX_SEATS];
        h[0] = (1 << 5) | (1 << 18); // suit0 rank5, suit1 rank5
        h[1] = (1 << 0) | (1 << 1); // suit0 rank0, suit0 rank1
        h[2] = (1 << 6) | (1 << 30); // suit0 rank6, suit2 rank4
        h
    }

    #[test]
    fn toy_config_is_enumerable() {
        let leaves = count_playout_leaves(&deal_3x2(), 3, 2);
        assert!(leaves > 0);
        assert!(leaves < 100_000, "tree unexpectedly large: {leaves}");
    }

    #[test]
    fn dd_tricks_sum_to_cards() {
        let hands = deal_3x2();
        let bids = [0, 0, 0, 0, 0, 0];
        let res = dd_play(&hands, &bids, ScoringConfig::default(), 3, 2);
        let total: u32 = res.tricks.iter().map(|&t| t as u32).sum();
        assert_eq!(total, 2);
        // Sanity: seat 2 holds the suit-0 ace-equivalent (rank 6) so it should
        // be able to win the suit-0 trick.
        assert!(res.tricks[2] >= 1);
    }

    #[test]
    fn dd_deterministic() {
        let hands = deal_3x2();
        let bids = [0, 0, 0, 0, 0, 0];
        let a = dd_play(&hands, &bids, ScoringConfig::default(), 3, 2);
        let b = dd_play(&hands, &bids, ScoringConfig::default(), 3, 2);
        assert_eq!(a.scores, b.scores);
        assert_eq!(a.tricks, b.tricks);
    }
}
