//! Single-game round driver (§3.2, §3.3, §3.4).
//!
//! `RoundState` plays one complete round: deal → bidding → play → scoring.
//! It is driven one decision at a time (`step(action)`), which is the shape
//! the lockstep batch needs later: each game records the acting seat in
//! `actor` (absolute seat) and advances by *decision count*, not by seat
//! identity. In debug builds the round invariant is asserted at the end:
//! every game consumes exactly `P + P*C` decisions.

use crate::cards::{bit, suit, Card, CardSet, FULL_DECK, MAX_SEATS};
use crate::knowledge::PublicKnowledge;
use crate::legal::{legal_bids, legal_plays};
use crate::prng::Rng;
use crate::scoring::ScoringConfig;

/// Phase of a round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Bidding,
    Playing,
    Done,
}

/// One round of Plump for `P` players with `C` cards each.
#[derive(Clone, Debug)]
pub struct RoundState {
    pub n_players: u8,
    pub n_cards: u8,
    pub dealer: u8,
    pub scoring: ScoringConfig,
    /// Current cards in each seat's hand.
    pub hands: [CardSet; MAX_SEATS],
    /// Bid made by each seat (meaningful only where `has_bid` is set).
    pub bids: [u8; MAX_SEATS],
    pub has_bid: [bool; MAX_SEATS],
    pub tricks_won: [u8; MAX_SEATS],
    /// Cards laid in the current trick, by play order (leader first).
    pub trick_cards: [Card; MAX_SEATS],
    /// Seat that laid each card of the current trick.
    pub trick_seats: [u8; MAX_SEATS],
    pub trick_count: u8,
    pub trick_leader: u8,
    /// Led suit of the current trick (set once the leader plays).
    pub led: Option<u8>,
    pub tricks_played: u8,
    pub phase: Phase,
    /// Absolute seat to act at the next `step`.
    pub actor: u8,
    pub steps_taken: usize,
    pub public: PublicKnowledge,
}

impl RoundState {
    /// Deals a fresh round with the given seed and starts bidding.
    pub fn new(n_players: u8, n_cards: u8, dealer: u8, seed: u64, scoring: ScoringConfig) -> Self {
        assert!(
            n_players >= 3 && n_players as usize <= MAX_SEATS,
            "bad n_players"
        );
        assert!(n_cards >= 1, "bad n_cards");
        assert!(n_players as u16 * n_cards as u16 <= 52, "deck too small");
        assert!(dealer < n_players, "bad dealer");
        let mut g = Self {
            n_players,
            n_cards,
            dealer,
            scoring,
            hands: [0; MAX_SEATS],
            bids: [0; MAX_SEATS],
            has_bid: [false; MAX_SEATS],
            tricks_won: [0; MAX_SEATS],
            trick_cards: [0; MAX_SEATS],
            trick_seats: [0; MAX_SEATS],
            trick_count: 0,
            trick_leader: 0,
            led: None,
            tricks_played: 0,
            phase: Phase::Bidding,
            actor: (dealer + 1) % n_players,
            steps_taken: 0,
            public: PublicKnowledge::empty(),
        };
        let mut rng = Rng::new(seed);
        g.deal(&mut rng);
        g
    }

    /// Round over a pre-dealt set of hands (tests / search ground truth).
    pub fn from_deal(
        n_players: u8,
        n_cards: u8,
        dealer: u8,
        scoring: ScoringConfig,
        hands: [CardSet; MAX_SEATS],
    ) -> Self {
        let mut g = Self::new(n_players, n_cards, dealer, 0, scoring);
        g.hands = hands;
        g.public = PublicKnowledge::new(n_players, n_cards, &g.hands);
        g
    }

    fn deal(&mut self, rng: &mut Rng) {
        let mut deck: [Card; 52] = core::array::from_fn(|i| i as u8);
        rng.shuffle(&mut deck);
        let p = self.n_players as usize;
        let c = self.n_cards as usize;
        for seat in 0..p {
            let mut h: CardSet = 0;
            for i in 0..c {
                h |= bit(deck[i * p + seat]);
            }
            self.hands[seat] = h;
        }
        self.public = PublicKnowledge::new(self.n_players, self.n_cards, &self.hands);
    }

    /// 11-bit mask over bids `0..=n_cards` legal for `seat` right now.
    pub fn legal_bid_mask(&self, seat: u8) -> u16 {
        let mut sum_others: u8 = 0;
        for p in 0..self.n_players {
            if p != seat && self.has_bid[p as usize] {
                sum_others += self.bids[p as usize];
            }
        }
        legal_bids(self.n_cards, sum_others, seat == self.dealer)
    }

    /// 52-bit mask over plays legal for `seat` right now.
    pub fn legal_play_mask(&self, seat: u8) -> CardSet {
        legal_plays(self.hands[seat as usize], self.led)
    }

    /// Applies one decision from the current actor (§3.2). In debug builds
    /// asserts the action is legal and that `PublicKnowledge` stays valid.
    pub fn step(&mut self, action: u8) {
        debug_assert!(self.phase != Phase::Done, "step after round end");
        let actor = self.actor;
        match self.phase {
            Phase::Bidding => self.apply_bid(actor, action),
            Phase::Playing => self.apply_play(actor, action),
            Phase::Done => unreachable!(),
        }
        self.steps_taken += 1;
        debug_assert!(
            self.public
                .is_valid(self.n_players as usize, self.n_cards as usize),
            "PublicKnowledge invariant violated"
        );
        if self.phase == Phase::Done {
            let d = self.n_players as usize * (1 + self.n_cards as usize);
            debug_assert_eq!(self.steps_taken, d, "round length invariant");
        }
    }

    fn apply_bid(&mut self, seat: u8, bid: u8) {
        let mask = self.legal_bid_mask(seat);
        debug_assert!(
            mask & (1u16 << bid) != 0,
            "illegal bid {bid} for seat {seat}"
        );
        self.bids[seat as usize] = bid;
        self.has_bid[seat as usize] = true;
        if self.has_bid[..self.n_players as usize].iter().all(|&b| b) {
            self.phase = Phase::Playing;
            self.trick_leader = (self.dealer + 1) % self.n_players;
            self.actor = self.trick_leader;
        } else {
            self.actor = (seat + 1) % self.n_players;
        }
    }

    fn apply_play(&mut self, seat: u8, card: Card) {
        let mask = self.legal_play_mask(seat);
        debug_assert!(mask & bit(card) != 0, "illegal play {card} for seat {seat}");
        self.hands[seat as usize] &= !bit(card);
        let idx = self.trick_count as usize;
        self.trick_cards[idx] = card;
        self.trick_seats[idx] = seat;
        self.trick_count += 1;
        if self.trick_count == 1 {
            self.led = Some(suit(card));
        }
        let led = self.led;
        self.public.on_play(seat, card, led);
        if self.trick_count == self.n_players {
            self.resolve_trick();
        } else {
            self.actor = (seat + 1) % self.n_players;
        }
    }

    fn resolve_trick(&mut self) {
        let led_suit = suit(self.trick_cards[0]);
        let mut winner = self.trick_seats[0];
        let mut best = self.trick_cards[0];
        for i in 1..self.n_players as usize {
            let c = self.trick_cards[i];
            if suit(c) == led_suit && c > best {
                best = c;
                winner = self.trick_seats[i];
            }
        }
        self.tricks_won[winner as usize] += 1;
        self.tricks_played += 1;
        self.trick_count = 0;
        self.led = None;
        if self.tricks_played == self.n_cards {
            self.phase = Phase::Done;
        } else {
            self.trick_leader = winner;
            self.actor = winner;
        }
    }

    /// Round score for each seat (§4.1 terminal per-seat return).
    pub fn round_scores(&self) -> [i32; MAX_SEATS] {
        let mut s = [0; MAX_SEATS];
        for (p, slot) in s.iter_mut().enumerate().take(self.n_players as usize) {
            *slot = self.scoring.score(self.bids[p], self.tricks_won[p]);
        }
        s
    }

    /// Actor-relative `possible[seat]` masks (§3.3): the actor's own hand is
    /// exact; others may hold anything not in the actor's hand, not played,
    /// and not voided for them.
    pub fn possible_for(&self, actor: u8) -> [CardSet; MAX_SEATS] {
        let mut out = [0; MAX_SEATS];
        let own = self.hands[actor as usize];
        for (p, slot) in out.iter_mut().enumerate().take(self.n_players as usize) {
            if p as u8 == actor {
                *slot = own;
            } else {
                *slot = FULL_DECK & !own & !self.public.all_played & !self.public.voids[p];
            }
        }
        out
    }

    pub fn is_done(&self) -> bool {
        self.phase == Phase::Done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deal_dims_and_dead_stock() {
        let g = RoundState::new(5, 10, 0, 1, ScoringConfig::default());
        for p in 0..5 {
            assert_eq!(g.hands[p].count_ones(), 10);
        }
        let dealt: CardSet = g.hands.iter().fold(0, |acc, h| acc | h);
        assert_eq!(dealt.count_ones(), 50);
        assert_eq!(g.public.undealt_count, 2);
        assert_eq!(g.public.undealt_possible.count_ones(), 2);
        assert_eq!(g.public.undealt_possible & dealt, 0);
    }

    #[test]
    fn bidding_order_dealer_last_and_roles_rotate() {
        for dealer in 0..5u8 {
            let mut g = RoundState::new(5, 3, dealer, 2, ScoringConfig::default());
            let mut order = vec![];
            while !g.is_done() {
                let seat = g.actor;
                if g.phase == Phase::Playing {
                    break;
                }
                let mask = g.legal_bid_mask(seat);
                order.push(seat);
                let bid = mask.trailing_zeros() as u8;
                g.step(bid);
            }
            assert_eq!(order.len(), 5);
            assert_eq!(
                order,
                vec![
                    (dealer + 1) % 5,
                    (dealer + 2) % 5,
                    (dealer + 3) % 5,
                    (dealer + 4) % 5,
                    dealer
                ]
            );
        }
    }

    #[test]
    fn full_round_invariant_and_scoring() {
        let n_players = 5u8;
        let n_cards = 3u8;
        for seed in 0..200u64 {
            let mut g = RoundState::new(
                n_players,
                n_cards,
                seed as u8 % n_players,
                seed,
                ScoringConfig::default(),
            );
            while !g.is_done() {
                let mask = match g.phase {
                    Phase::Bidding => g.legal_bid_mask(g.actor) as u64,
                    Phase::Playing => g.legal_play_mask(g.actor),
                    Phase::Done => unreachable!(),
                };
                assert_ne!(mask, 0, "empty legal mask");
                let action = mask.trailing_zeros() as u8;
                g.step(action);
            }
            assert_eq!(g.steps_taken, n_players as usize * (1 + n_cards as usize));
            for p in 0..n_players as usize {
                assert_eq!(g.hands[p], 0, "hands not empty");
            }
            let total_tricks: u32 = g.tricks_won.iter().map(|&t| t as u32).sum();
            assert_eq!(total_tricks, n_cards as u32);
            // Rewards must equal the scoring function applied to (bids, tricks).
            let scores = g.round_scores();
            for (p, &s) in scores.iter().enumerate().take(n_players as usize) {
                assert_eq!(s, g.scoring.score(g.bids[p], g.tricks_won[p]));
            }
            assert!(g.public.is_valid(n_players as usize, n_cards as usize));
        }
    }

    #[test]
    fn random_walk_plays_legally_and_validates_knowledge() {
        let mut g = RoundState::new(5, 4, 2, 9, ScoringConfig::default());
        let mut rng = Rng::new(99);
        while !g.is_done() {
            let (mask, n_actions) = match g.phase {
                Phase::Bidding => (g.legal_bid_mask(g.actor) as u64, g.n_cards as usize + 1),
                Phase::Playing => (g.legal_play_mask(g.actor), 52),
                Phase::Done => unreachable!(),
            };
            // pick a uniformly random legal bit
            let legal_bits: Vec<u32> = (0..n_actions)
                .filter(|&i| mask & (1u64 << i) != 0)
                .map(|i| i as u32)
                .collect();
            let pick = legal_bits[rng.next_below(legal_bits.len())];
            g.step(pick as u8);
        }
        assert_eq!(g.steps_taken, 5 * (1 + 4));
    }

    #[test]
    fn trick_winner_follow_suit_and_void() {
        // Deal so seat 2 holds the high card of suit 0.
        let mut hands = [0u64; MAX_SEATS];
        hands[0] = (1 << 5) | (1 << 18); // suit0 r5, suit1 r5
        hands[1] = (1 << 0) | (1 << 1); // suit0 r0, suit0 r1
        hands[2] = (1 << 6) | (1 << 30); // suit0 r6, suit2 r4
        let mut g = RoundState::from_deal(3, 2, 2, ScoringConfig::default(), hands);
        // Bids: all 0 (dealer=2 cannot bid 2; bid 0 is legal).
        for _ in 0..3 {
            let seat = g.actor;
            g.step(0);
            assert!(g.has_bid[seat as usize]);
        }
        assert_eq!(g.phase, Phase::Playing);
        // Trick 1, seat 0 leads suit 0.
        assert_eq!(g.actor, 0);
        g.step(5); // s0 leads suit0
        assert_eq!(g.led, Some(0));
        g.step(1); // s1 follows suit0
        g.step(6); // s2 follows suit0, wins with rank 6
        assert_eq!(g.tricks_won[2], 1);
        assert_eq!(g.tricks_played, 1);
        assert_eq!(g.actor, 2); // winner leads next
                                // Trick 2, seat 2 leads suit 2 (card 30). Seat 0 is void in suit 2.
        g.step(30);
        g.step(18); // s0: void -> free choice
        g.step(0); // s1: void -> free choice
        assert_eq!(g.tricks_won[2], 2);
        assert_eq!(g.tricks_won[0], 0);
        assert_eq!(g.tricks_won[1], 0);
        assert!(g.is_done());
        assert_eq!(g.round_scores()[0], 5); // made 0-bid
        assert_eq!(g.round_scores()[1], 5);
        assert_eq!(g.round_scores()[2], 0); // bid 0, took 2: miss
        assert_eq!(g.steps_taken, 3 * (1 + 2));
    }

    #[test]
    fn possible_for_is_actor_relative() {
        use crate::cards::SUIT_MASK;
        let mut hands = [0u64; MAX_SEATS];
        hands[0] = (1 << 0) | (1 << 13); // suit0 r0, suit1 r0
        hands[1] = (1 << 1) | (1 << 14); // suit0 r1, suit1 r1
        hands[2] = (1 << 15) | (1 << 16); // both suit 1: seat 2 is void in suit 0
                                          // dealer = 2 so seat 0 leads the first trick.
        let mut g = RoundState::from_deal(3, 2, 2, ScoringConfig::default(), hands);
        for _ in 0..3 {
            g.step(0);
        }
        // Play: seat 0 leads card 0 (suit 0); seat 1 follows with 1; seat 2
        // has no suit 0 -> plays 15 and is proven void in suit 0.
        g.step(0);
        g.step(1);
        g.step(15);
        assert_eq!(g.public.voids[2], SUIT_MASK[0]);
        // From seat 1's perspective: own hand exact; others exclude own hand,
        // played cards, and proven voids.
        let poss = g.possible_for(1);
        assert_eq!(poss[1], g.hands[1]);
        for (p, slot) in poss.iter().enumerate().take(3) {
            if p == 1 {
                continue;
            }
            assert_eq!(slot & g.hands[1], 0, "other may not hold my cards");
            assert_eq!(slot & g.public.all_played, 0, "played cards excluded");
            assert_eq!(slot & g.public.voids[p], 0, "voided suit excluded");
        }
        // Seat 1 played card 1, so its remaining hand is exactly card 14.
        assert_eq!(poss[1], (1 << 14));
    }
}
