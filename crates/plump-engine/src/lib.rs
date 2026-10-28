//! Plump engine: a compact `u64` bitboard implementation of the Swedish
//! card game Plump, plus PIMC/endgame search, suit-isomorphism augmentation
//! and a parallel vectorised environment for reinforcement learning.

pub mod augment;
pub mod bots;
pub mod engine;
pub mod search;
pub mod vec_env;

#[cfg(feature = "python")]
pub mod python;

pub use bots::{
    bot_action, final_boss_card, regular_bid, regular_card, rookie_bid, rookie_card,
    tactician_bid, tactician_card, BOT_FINAL_BOSS, BOT_REGULAR, BOT_ROOKIE, BOT_TACTICIAN,
};
pub use engine::{
    card_index, display_points, rank_of, suit_of, terminal_reward, FastEnv, Phase, BID_BASE,
    CARD_BASE, FEAT_DIM, MAX_SEQ, PAD_TOK, PLAYER_BASE, ROUND_SCHEDULE, VOCAB,
};
