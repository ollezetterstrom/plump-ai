//! Feature encoder (§3.6).
//!
//! Phase 2 material. This module declares the performance contract and the
//! fixed-width batch shape; the bit-iteration encoder, rayon parallelism, and
//! double-buffered pipelining land with the game state.

/// Hard cap on active features per row, so the host→device copy is a single
/// contiguous `[B, K_MAX]` transfer with static shape.
pub const K_MAX: usize = 128;
