//! Public-knowledge state (§3.3).
//!
//! The engine maintains, per game, hard constraint structures that *any* world
//! sampler must respect. Belief logits may reweight the support but can never
//! expand it. The struct holds the **globally observable** facts; the
//! actor-relative `possible` masks are computed on demand by the round driver
//! (`RoundState::possible_for`), which knows the actor's own hand.

use crate::cards::{bit, Card, FULL_DECK};
use crate::cards::{suit, CardSet, MAX_SEATS, SUIT_MASK};

/// Hard public-knowledge constraints for one game state.
#[derive(Clone, Copy, Debug)]
pub struct PublicKnowledge {
    /// Remaining hand size per seat.
    pub remaining_hand_size: [u8; MAX_SEATS],
    /// `52 - P*C`: dead-stock size, constant per round.
    pub undealt_count: u8,
    /// Cards that could still be in the dead stock (ground truth at deal).
    pub undealt_possible: CardSet,
    /// Cards played so far by each seat.
    pub played_by: [CardSet; MAX_SEATS],
    /// Union of all played cards.
    pub all_played: CardSet,
    /// Suits each seat has *proven* void in (failed to follow suit).
    pub voids: [CardSet; MAX_SEATS],
}

impl PublicKnowledge {
    /// Fresh knowledge after a deal of `n_cards` to each of `n_players`
    /// players, given the (ground-truth) dealt hands.
    pub fn new(n_players: u8, n_cards: u8, hands: &[CardSet; MAX_SEATS]) -> Self {
        let mut pk = Self::empty();
        pk.undealt_count = 52 - n_players * n_cards;
        pk.undealt_possible = FULL_DECK;
        for (p, size) in pk
            .remaining_hand_size
            .iter_mut()
            .enumerate()
            .take(n_players as usize)
        {
            *size = n_cards;
            pk.undealt_possible &= !hands[p];
        }
        pk
    }

    /// An empty (invalid) instance; games populate it incrementally during
    /// the deal and on every play.
    pub const fn empty() -> Self {
        Self {
            remaining_hand_size: [0; MAX_SEATS],
            undealt_count: 0,
            undealt_possible: 0,
            played_by: [0; MAX_SEATS],
            all_played: 0,
            voids: [0; MAX_SEATS],
        }
    }

    /// Incremental update on every play (§3.3):
    /// - the played card leaves every seat's reach and the dead stock;
    /// - a failed follow-suit proves a void (`SUIT_MASK[s]` cleared).
    pub fn on_play(&mut self, seat: u8, card: Card, led: Option<u8>) {
        self.played_by[seat as usize] |= bit(card);
        self.all_played |= bit(card);
        self.remaining_hand_size[seat as usize] -= 1;
        self.undealt_possible &= !bit(card);
        if let Some(s) = led {
            if suit(card) != s {
                self.voids[seat as usize] |= SUIT_MASK[s as usize];
            }
        }
    }

    /// Structural invariants that hold at every state of a valid round.
    /// In debug builds the round driver asserts this after every step.
    pub fn is_valid(&self, n_players: usize, n_cards: usize) -> bool {
        if n_players == 0 || n_players > MAX_SEATS {
            return false;
        }
        if self.undealt_count as usize != 52 - n_players * n_cards {
            return false;
        }
        let sum_hands: usize = self.remaining_hand_size[..n_players]
            .iter()
            .map(|&h| h as usize)
            .sum();
        // Hands + dead stock must account for every unseen card.
        sum_hands + self.undealt_count as usize == 52 - self.all_played.count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_invalid_for_any_real_round() {
        assert!(!PublicKnowledge::empty().is_valid(5, 10));
    }

    #[test]
    fn fresh_deal_is_valid() {
        let mut hands = [0u64; MAX_SEATS];
        hands[0] = (1 << 0) | (1 << 13);
        hands[1] = (1 << 1) | (1 << 14);
        hands[2] = (1 << 2) | (1 << 15);
        let pk = PublicKnowledge::new(3, 2, &hands);
        assert!(pk.is_valid(3, 2));
        assert_eq!(pk.undealt_count, 52 - 6);
        assert_eq!(pk.undealt_possible.count_ones(), 46);
    }

    #[test]
    fn on_play_detects_void() {
        let mut hands = [0u64; MAX_SEATS];
        hands[0] = (1 << 0) | (1 << 13);
        hands[1] = (1 << 1) | (1 << 14);
        hands[2] = (1 << 2) | (1 << 15);
        let mut pk = PublicKnowledge::new(3, 2, &hands);
        // Seat 1 plays card 1 (suit 0) while suit 0 is led -> follows.
        pk.on_play(1, 1, Some(0));
        assert_eq!(pk.voids[1], 0);
        // Seat 2 plays card 15 (suit 1) while suit 0 is led -> void in suit 0.
        pk.on_play(2, 15, Some(0));
        assert_eq!(pk.voids[2], SUIT_MASK[0]);
        // Played cards leave all reachable sets.
        assert_eq!(pk.all_played.count_ones(), 2);
        assert_eq!(pk.remaining_hand_size[1], 1);
        assert_eq!(pk.remaining_hand_size[2], 1);
    }
}
