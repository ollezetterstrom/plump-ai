//! Plump engine: a compact `u64` bitboard implementation of the Swedish
//! card game Plump, plus PIMC/endgame search, suit-isomorphism augmentation
//! and a parallel vectorised environment for reinforcement learning.

pub mod augment;
pub mod engine;
pub mod search;
pub mod vec_env;

#[cfg(feature = "python")]
pub mod python;

pub use engine::{
    card_index, display_points, rank_of, suit_of, terminal_reward, FastEnv, Phase, BID_BASE,
    CARD_BASE, FEAT_DIM, MAX_SEQ, PAD_TOK, PLAYER_BASE, ROUND_SCHEDULE, VOCAB,
};
