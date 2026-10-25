//! Parallel vectorised environment for lockstep GPU training.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::engine::{
    display_points, terminal_reward, FastEnv, Phase, FEAT_DIM, MAX_SEQ, ROUND_SCHEDULE,
};

/// One observation row per environment (the seat that must act next).
pub struct BatchObs {
    pub num_envs: usize,
    pub toks: Vec<i64>,       // [N, 96]
    pub feats: Vec<f32>,      // [N, 296]
    pub legal_masks: Vec<u8>, // [N, 52], bid bits use 0..=10
    pub is_bid_phase: Vec<bool>,
    pub current_players: Vec<u8>,
}

impl BatchObs {
    fn new(n: usize) -> Self {
        Self {
            num_envs: n,
            toks: vec![0; n * MAX_SEQ],
            feats: vec![0.0; n * FEAT_DIM],
            legal_masks: vec![0; n * 52],
            is_bid_phase: vec![false; n],
            current_players: vec![0; n],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Transition {
    pub toks: [i64; MAX_SEQ],
    pub feats: [f32; FEAT_DIM],
    pub opp_hands: [f32; 156],
    pub action: u8,
    pub ret: f32,
    pub final_tricks: u8,
    pub player: u8,
    pub is_play: bool,
    pub env_idx: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct RoundSummary {
    pub env_idx: usize,
    pub round_cards: u8,
    pub start_bidder: u8,
    pub bids: [i8; 4],
    pub tricks_won: [u8; 4],
    pub points: [i32; 4],
    pub rewards: [f32; 4],
}

pub struct StepOutput {
    pub transitions: Vec<Transition>,
    pub summaries: Vec<RoundSummary>,
}

#[derive(Clone)]
struct PendingTransition {
    toks: [i64; MAX_SEQ],
    feats: [f32; FEAT_DIM],
    opp_hands: [f32; 156],
    action: u8,
    player: u8,
    is_play: bool,
    trick: u8,
}

/// A bank of independent Plump rounds advanced in lockstep.
pub struct RustVecEnv {
    pub envs: Vec<FastEnv>,
    schedule_pos: Vec<usize>,
    pending: Vec<Vec<PendingTransition>>,
    rng: StdRng,
    pub rounds_completed: u64,
}

impl RustVecEnv {
    pub fn new(num_envs: usize, seed: u64) -> Self {
        assert!(num_envs > 0);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut envs = Vec::with_capacity(num_envs);
        let mut schedule_pos = Vec::with_capacity(num_envs);
        for i in 0..num_envs {
            let pos = i % ROUND_SCHEDULE.len();
            let start = (i % 4) as u8;
            let mut env = FastEnv::default();
            env.new_round(ROUND_SCHEDULE[pos], start, &mut rng);
            envs.push(env);
            schedule_pos.push(pos);
        }
        let pending = (0..num_envs).map(|_| Vec::with_capacity(48)).collect();
        Self {
            envs,
            schedule_pos,
            pending,
            rng,
            rounds_completed: 0,
        }
    }

    #[inline]
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    #[inline]
    fn legal_mask_52(env: &FastEnv) -> [u8; 52] {
        let mut out = [0u8; 52];
        match env.phase {
            Phase::Bidding => {
                let mask = env.legal_bids_mask(env.current_player as usize);
                for bid in 0..=10 {
                    if mask & (1u16 << bid) != 0 {
                        out[bid] = 1;
                    }
                }
            }
            Phase::Playing => {
                let mask = env.legal_cards_mask(env.current_player as usize);
                for card in 0..52 {
                    if mask & (1u64 << card) != 0 {
                        out[card] = 1;
                    }
                }
            }
            Phase::Terminal => {}
        }
        out
    }

    /// Encode the current decision of every environment, in parallel.
    pub fn observe_batch(&self) -> BatchObs {
        let n = self.envs.len();
        let rows: Vec<([i64; MAX_SEQ], [f32; FEAT_DIM], [u8; 52], bool, u8)> = self
            .envs
            .par_iter()
            .map(|env| {
                let p = env.current_player as usize;
                let mut toks = [0i64; MAX_SEQ];
                let mut feats = [0.0f32; FEAT_DIM];
                let mut opp = [0.0f32; 156];
                env.encode_decision(p, &mut toks, &mut feats, &mut opp);
                (
                    toks,
                    feats,
                    Self::legal_mask_52(env),
                    env.phase == Phase::Bidding,
                    env.current_player,
                )
            })
            .collect();

        let mut obs = BatchObs::new(n);
        for (i, (toks, feats, legal, is_bid, player)) in rows.into_iter().enumerate() {
            obs.toks[i * MAX_SEQ..(i + 1) * MAX_SEQ].copy_from_slice(&toks);
            obs.feats[i * FEAT_DIM..(i + 1) * FEAT_DIM].copy_from_slice(&feats);
            obs.legal_masks[i * 52..(i + 1) * 52].copy_from_slice(&legal);
            obs.is_bid_phase[i] = is_bid;
            obs.current_players[i] = player;
        }
        obs
    }

    fn reset_env(&mut self, i: usize) {
        self.schedule_pos[i] = (self.schedule_pos[i] + 1) % ROUND_SCHEDULE.len();
        let next_start = (self.envs[i].start_bidder + 1) % 4;
        let round_cards = ROUND_SCHEDULE[self.schedule_pos[i]];
        self.envs[i].new_round(round_cards, next_start, &mut self.rng);
        self.pending[i].clear();
    }

    /// Advance every environment by one decision. Completed rounds are
    /// finalised into training transitions and the environments are reset.
    pub fn step_batch(&mut self, actions: &[u8]) -> StepOutput {
        assert_eq!(
            actions.len(),
            self.envs.len(),
            "one action per environment is required"
        );

        let mut output = StepOutput {
            transitions: Vec::new(),
            summaries: Vec::new(),
        };

        for i in 0..self.envs.len() {
            if self.envs[i].phase == Phase::Terminal {
                self.reset_env(i);
            }

            let env = &mut self.envs[i];
            let player = env.current_player as usize;
            let is_play = env.phase == Phase::Playing;
            let mut toks = [0i64; MAX_SEQ];
            let mut feats = [0.0f32; FEAT_DIM];
            let mut opp = [0.0f32; 156];
            env.encode_decision(player, &mut toks, &mut feats, &mut opp);

            debug_assert!(
                Self::legal_mask_52(env)[actions[i] as usize] != 0,
                "illegal action {} in env {} ({:?})",
                actions[i],
                i,
                env.phase
            );

            self.pending[i].push(PendingTransition {
                toks,
                feats,
                opp_hands: opp,
                action: actions[i],
                player: player as u8,
                is_play,
                trick: env.current_trick,
            });

            match env.phase {
                Phase::Bidding => env.step_bid(actions[i]),
                Phase::Playing => {
                    env.step_play(actions[i]);
                }
                Phase::Terminal => {}
            }

            if env.phase == Phase::Terminal {
                let (transitions, summary) = self.finalize_env(i);
                output.transitions.extend(transitions);
                output.summaries.push(summary);
                self.rounds_completed += 1;
                self.reset_env(i);
            }
        }

        output
    }

    /// Compute the round's bid/play targets (true cumulative DMC returns)
    /// and drain the pending transitions.
    fn finalize_env(&mut self, i: usize) -> (Vec<Transition>, RoundSummary) {
        let env = self.envs[i];
        let round_cards = env.round_cards as usize;
        let winners = env.trick_winners();

        // Per-trick shaped reward for every player, using cumulative tricks.
        let mut shaped = vec![[0.0f32; 4]; round_cards];
        let mut cumulative = [0u8; 4];
        for t in 0..round_cards {
            if let Some(&w) = winners.get(t) {
                cumulative[w] += 1;
            }
            for p in 0..4 {
                let bid = env.bids[p].max(0) as i32;
                let won = winners.get(t).copied() == Some(p);
                let count = cumulative[p] as i32;
                shaped[t][p] = if won {
                    if count <= bid {
                        1.0
                    } else {
                        -1.5
                    }
                } else if count < bid {
                    -1.0
                } else {
                    0.5
                };
            }
        }
        let mut suffix = vec![[0.0f32; 4]; round_cards + 1];
        for t in (0..round_cards).rev() {
            for p in 0..4 {
                suffix[t][p] = suffix[t + 1][p] + shaped[t][p];
            }
        }

        let terminal: [f32; 4] =
            std::array::from_fn(|p| terminal_reward(env.tricks_won[p], env.bids[p]));

        let mut transitions = Vec::with_capacity(self.pending[i].len());
        for rec in self.pending[i].drain(..) {
            let p = rec.player as usize;
            let ret = if rec.is_play {
                let t = (rec.trick as usize).min(round_cards);
                terminal[p] + 0.20 * suffix[t][p]
            } else {
                terminal[p]
            };
            transitions.push(Transition {
                toks: rec.toks,
                feats: rec.feats,
                opp_hands: rec.opp_hands,
                action: rec.action,
                ret,
                final_tricks: env.tricks_won[p],
                player: rec.player,
                is_play: rec.is_play,
                env_idx: i,
            });
        }

        let summary = RoundSummary {
            env_idx: i,
            round_cards: env.round_cards,
            start_bidder: env.start_bidder,
            bids: env.bids,
            tricks_won: env.tricks_won,
            points: std::array::from_fn(|p| display_points(env.tricks_won[p], env.bids[p])),
            rewards: terminal,
        };
        (transitions, summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_batch_shapes_and_legal_actions() {
        let env = RustVecEnv::new(8, 1);
        let obs = env.observe_batch();
        assert_eq!(obs.toks.len(), 8 * MAX_SEQ);
        assert_eq!(obs.feats.len(), 8 * FEAT_DIM);
        assert_eq!(obs.legal_masks.len(), 8 * 52);
        assert_eq!(obs.current_players.len(), 8);
        for i in 0..8 {
            assert_eq!(obs.toks[i * MAX_SEQ], 1);
            assert!(obs.is_bid_phase[i]);
            let row = &obs.legal_masks[i * 52..(i + 1) * 52];
            // Each first bidder can bid the full 0..=round_cards range.
            assert!(row[0] == 1);
            assert!(row.iter().take(11).any(|&v| v == 1));
        }
    }

    #[test]
    fn full_rounds_emit_all_player_transitions_and_reset() {
        let mut env = RustVecEnv::new(4, 42);
        let mut summaries = Vec::new();
        let mut transitions = Vec::new();
        let mut steps = 0;
        while summaries.len() < 4 && steps < 500 {
            let obs = env.observe_batch();
            let mut actions = vec![0u8; 4];
            for i in 0..4 {
                let row = &obs.legal_masks[i * 52..(i + 1) * 52];
                let action = row.iter().position(|&v| v == 1).unwrap() as u8;
                actions[i] = action;
            }
            let out = env.step_batch(&actions);
            summaries.extend(out.summaries);
            transitions.extend(out.transitions);
            steps += 1;
        }
        assert!(summaries.len() >= 4, "rounds did not finish");
        assert_eq!(env.rounds_completed as usize, summaries.len());

        for summary in &summaries {
            let count = transitions
                .iter()
                .filter(|t| t.env_idx == summary.env_idx && t.is_play)
                .count();
            let bids = transitions
                .iter()
                .filter(|t| t.env_idx == summary.env_idx && !t.is_play)
                .count();
            assert!(count > 0);
            assert!(bids >= 4);
            assert_eq!(
                summary.tricks_won.iter().map(|&x| x as u32).sum::<u32>(),
                summary.round_cards as u32
            );
            for t in transitions.iter() {
                assert!(t.ret.is_finite());
                assert!(t.final_tricks <= 10);
            }
        }
    }

    #[test]
    fn terminal_returns_follow_scoring() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(11);
        let mut round = FastEnv::default();
        round.new_round(1, 0, &mut rng);
        while round.phase == Phase::Bidding {
            let p = round.current_player as usize;
            let mask = round.legal_bids_mask(p);
            round.step_bid(mask.trailing_zeros() as u8);
        }
        while round.phase == Phase::Playing {
            let p = round.current_player as usize;
            let mask = round.legal_cards_mask(p);
            round.step_play(mask.trailing_zeros() as u8);
        }
        assert_eq!(round.phase, Phase::Terminal);

        let mut vec = RustVecEnv::new(1, 1);
        vec.envs[0] = round;
        let blank = PendingTransition {
            toks: [0i64; MAX_SEQ],
            feats: [0.0f32; FEAT_DIM],
            opp_hands: [0.0f32; 156],
            action: 0,
            player: 0,
            is_play: true,
            trick: 0,
        };
        // Four bid records and four play records for the one trick.
        vec.pending[0] = (0..4)
            .map(|p| PendingTransition {
                player: p as u8,
                is_play: false,
                ..blank.clone()
            })
            .chain((0..4).map(|p| PendingTransition {
                player: p as u8,
                is_play: true,
                ..blank.clone()
            }))
            .collect();

        let (transitions, summary) = vec.finalize_env(0);
        assert_eq!(transitions.len(), 8);
        let winner = round.trick_winners()[0];
        for t in &transitions {
            let p = t.player as usize;
            let terminal = summary.rewards[p];
            if t.is_play {
                let shaped = if p == winner {
                    // Bid 0/1 matters: the bid in this test is the lowest
                    // legal one, which is 0 for the first three bidders.
                    let after = 1i32;
                    let bid = round.bids[p].max(0) as i32;
                    if after <= bid {
                        1.0
                    } else {
                        -1.5
                    }
                } else {
                    let bid = round.bids[p].max(0) as i32;
                    if 0 < bid {
                        -1.0
                    } else {
                        0.5
                    }
                };
                assert!((t.ret - (terminal + 0.20 * shaped)).abs() < 1e-6);
            } else {
                assert!((t.ret - terminal).abs() < 1e-6);
            }
        }
    }
}
