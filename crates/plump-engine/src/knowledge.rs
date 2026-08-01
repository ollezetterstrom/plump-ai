//! Public-knowledge state (§3.3).
//!
//! The engine maintains, per game, hard constraint structures that *any*
//! world sampler must respect. Belief logits may reweight the support but can
//! never expand it. This struct is Phase 1 material; only the shape and the
//! invariants are declared here.

use crate::cards::CardSet;

/// Maximum supported number of players (one per relative seat plus the
/// actor's own slot, with one unused slot).
pub const MAX_SEATS: usize = 6;

/// Hard public-knowledge constraints for one game state.
#[derive(Clone, Copy, Debug)]
pub struct PublicKnowledge {
    /// Cards seat `p` could still hold (voids, played, and own hand removed).
    pub possible: [CardSet; MAX_SEATS],
    /// Remaining hand size per seat.
    pub remaining_hand_size: [u8; MAX_SEATS],
    /// `52 - P*C`: dead-stock size, constant per round.
    pub undealt_count: u8,
    /// Cards that could still be in the dead stock.
    pub undealt_possible: CardSet,
    /// Cards played so far by each seat.
    pub played_by: [CardSet; MAX_SEATS],
    /// Union of all played cards.
    pub all_played: CardSet,
}

impl PublicKnowledge {
    /// An empty (invalid) instance; games populate it incrementally during
    /// the deal and on every play.
    pub const fn empty() -> Self {
        Self {
            possible: [0; MAX_SEATS],
            remaining_hand_size: [0; MAX_SEATS],
            undealt_count: 0,
            undealt_possible: 0,
            played_by: [0; MAX_SEATS],
            all_played: 0,
        }
    }

    /// Structural invariants that hold at every state of a valid round.
    /// In debug builds the world sampler re-checks every sampled world
    /// against the struct itself (`assert_world_valid`).
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
}
