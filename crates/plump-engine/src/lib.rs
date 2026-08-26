//! plump-engine — Rust rules + search, mirrors plump/env/engine.py
//! House v2: 0=5, earliest highest leads.
//! Decoupled: no Python, no torch. Used by plump/search/mcts.py for 10-50x rollout.

pub mod cards;
pub mod engine;
pub mod search;

pub use cards::{card_index, index_to_card, SUITS};
pub use engine::PlumpEnv;
