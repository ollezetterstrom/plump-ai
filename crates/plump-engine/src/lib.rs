//! `plump-engine`: the Rust game engine for Plump.
//!
//! Phase 0: card types, scoring config, legal masks, the feature layout
//! registry with its compile-time checks, and the structural suit-permutation
//! table.
//!
//! Phase 1: the single-game round driver (`game::RoundState`), incremental
//! `PublicKnowledge`, a deterministic PRNG, and the brute-force `P=3, C=2`
//! double-dummy reference (`search`).

pub mod cards;
pub mod encode;
pub mod game;
pub mod knowledge;
pub mod layout;
pub mod legal;
pub mod prng;
pub mod scoring;
pub mod search;

pub use cards::{Card, CardSet, FULL_DECK, MAX_SEATS, SUIT_MASK};
pub use encode::K_MAX;
pub use game::{Phase, RoundState};
pub use knowledge::PublicKnowledge;
pub use layout::{Block, SuitRule, N_FEATURES, N_PERMUTATIONS};
pub use prng::Rng;
pub use scoring::ScoringConfig;
pub use search::{dd_play, dd_scores, DdResult};
