//! `plump-engine`: the Rust game engine for Plump.
//!
//! Phase 0 (day 0) content: card types, scoring config, legal masks, the
//! feature layout registry with its compile-time checks, and the structural
//! suit-permutation table. Later phases add the round driver, `PublicKnowledge`
//! updates, the batch encoder, and the world sampler.

pub mod cards;
pub mod encode;
pub mod knowledge;
pub mod layout;
pub mod legal;
pub mod scoring;

pub use cards::{Card, CardSet, FULL_DECK, SUIT_MASK};
pub use encode::K_MAX;
pub use layout::{Block, SuitRule, N_FEATURES, N_PERMUTATIONS};
pub use scoring::ScoringConfig;
