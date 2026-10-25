//! PyO3 bindings for the Plump engine. Built with the `python` feature via
//! `maturin develop --release --features python`.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::augment::augment_batch;
use crate::engine::{display_points, terminal_reward, FastEnv, Phase, FEAT_DIM, MAX_SEQ};
use crate::search::{
    search_best_bid_pannplump, search_best_card, search_bid_values_pannplump, search_card_values,
};
use crate::vec_env::RustVecEnv;

fn cards_in(mask: u64) -> Vec<u8> {
    let mut cards = Vec::with_capacity(mask.count_ones() as usize);
    let mut m = mask;
    while m != 0 {
        cards.push(m.trailing_zeros() as u8);
        m &= m - 1;
    }
    cards
}

fn flat_i64<'py>(
    py: Python<'py>,
    data: Vec<i64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    data.into_pyarray_bound(py).reshape([rows, cols])
}

fn flat_f32<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    data.into_pyarray_bound(py).reshape([rows, cols])
}

fn flat_u8<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    data.into_pyarray_bound(py).reshape([rows, cols])
}

/// Single-round environment used by `play.py` and evaluation scripts.
#[pyclass(module = "plump_engine")]
pub struct PyFastEnv {
    env: FastEnv,
    rng: StdRng,
}

#[pymethods]
impl PyFastEnv {
    #[new]
    #[pyo3(signature = (seed = None))]
    fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            env: FastEnv::default(),
            rng,
        }
    }

    fn new_round(&mut self, round_cards: u8, start_bidder: u8) {
        self.env.new_round(round_cards, start_bidder, &mut self.rng);
    }

    /// `(tokens[96], features[296], opponent_hand_targets[156])`
    fn encode<'py>(
        &self,
        py: Python<'py>,
        viewer: usize,
    ) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
        let mut toks = [0i64; MAX_SEQ];
        let mut feats = [0.0f32; FEAT_DIM];
        let mut opp = [0.0f32; 156];
        self.env
            .encode_decision(viewer, &mut toks, &mut feats, &mut opp);
        Ok((
            toks.to_vec().into_pyarray_bound(py).unbind(),
            feats.to_vec().into_pyarray_bound(py).unbind(),
            opp.to_vec().into_pyarray_bound(py).unbind(),
        ))
    }

    fn legal_bids_mask(&self, player: usize) -> u16 {
        self.env.legal_bids_mask(player)
    }

    fn legal_cards_mask(&self, player: usize) -> u64 {
        self.env.legal_cards_mask(player)
    }

    fn legal_mask(&self) -> u64 {
        self.env.legal_mask()
    }

    fn step_bid(&mut self, bid: u8) {
        self.env.step_bid(bid);
    }

    fn step_play(&mut self, card_idx: u8) -> Option<usize> {
        self.env.step_play(card_idx)
    }

    fn get_leader(&self) -> usize {
        self.env.get_leader()
    }

    fn bid_offset(&self, player: usize) -> usize {
        self.env.bidding_offset(player)
    }

    fn forbidden_bid(&self) -> Option<u8> {
        let p = self.env.current_player as usize;
        if self.env.phase != Phase::Bidding || self.env.bidding_offset(p) != 3 {
            return None;
        }
        let sum: i32 = (0..4)
            .filter(|&q| q != p)
            .map(|q| self.env.bids[q] as i32)
            .sum();
        let forbidden = self.env.round_cards as i32 - sum;
        if (0..=self.env.round_cards as i32).contains(&forbidden) {
            Some(forbidden as u8)
        } else {
            None
        }
    }

    fn terminal_reward_for(&self, player: usize) -> f32 {
        terminal_reward(self.env.tricks_won[player], self.env.bids[player])
    }

    fn display_points_for(&self, player: usize) -> i32 {
        display_points(self.env.tricks_won[player], self.env.bids[player])
    }

    #[pyo3(signature = (n_worlds, belief_probs=None))]
    fn search_move<'py>(
        &self,
        py: Python<'py>,
        n_worlds: usize,
        belief_probs: Option<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<u8> {
        let belief = if let Some(array) = belief_probs {
            let view = array.as_array();
            if view.shape() != [3, 52] {
                return Err(PyValueError::new_err(format!(
                    "belief_probs must have shape (3, 52), got {:?}",
                    view.shape()
                )));
            }
            let mut probs = [[0.0f32; 52]; 3];
            for i in 0..3 {
                for j in 0..52 {
                    probs[i][j] = view[[i, j]];
                }
            }
            Some(probs)
        } else {
            None
        };
        let env = self.env;
        let viewer = env.current_player as usize;
        let card =
            py.allow_threads(move || search_best_card(&env, viewer, n_worlds, belief.as_ref()));
        Ok(card)
    }

    fn search_pannplump_bid(&self, py: Python<'_>) -> u8 {
        let env = self.env;
        let viewer = env.current_player as usize;
        py.allow_threads(move || search_best_bid_pannplump(&env, viewer, 49))
    }

    fn search_pannplump_bid_values(&self, py: Python<'_>) -> Vec<f32> {
        let env = self.env;
        let viewer = env.current_player as usize;
        py.allow_threads(move || search_bid_values_pannplump(&env, viewer, 49).to_vec())
    }

    /// Average PIMC terminal reward by card; illegal cards are `-inf`.
    #[pyo3(signature = (n_worlds, belief_probs=None))]
    fn search_card_values<'py>(
        &self,
        py: Python<'py>,
        n_worlds: usize,
        belief_probs: Option<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<Vec<f32>> {
        let belief = if let Some(array) = belief_probs {
            let view = array.as_array();
            if view.shape() != [3, 52] {
                return Err(PyValueError::new_err(format!(
                    "belief_probs must have shape (3, 52), got {:?}",
                    view.shape()
                )));
            }
            let mut probs = [[0.0f32; 52]; 3];
            for i in 0..3 {
                for j in 0..52 {
                    probs[i][j] = view[[i, j]];
                }
            }
            Some(probs)
        } else {
            None
        };
        let env = self.env;
        let viewer = env.current_player as usize;
        Ok(py.allow_threads(move || {
            search_card_values(&env, viewer, n_worlds, belief.as_ref()).to_vec()
        }))
    }

    fn state(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let hands: Vec<Vec<u8>> = self.env.hands.iter().map(|&m| cards_in(m)).collect();
        dict.set_item("hands", hands)?;
        dict.set_item("hand_masks", self.env.hands.to_vec())?;
        dict.set_item("played", self.env.played)?;
        dict.set_item("void_mask", self.env.void_mask.to_vec())?;
        dict.set_item("bids", self.env.bids.to_vec())?;
        dict.set_item("tricks_won", self.env.tricks_won.to_vec())?;
        let table: Vec<(u8, u8)> = self.env.table[..self.env.table_len as usize].to_vec();
        dict.set_item("table", table)?;
        dict.set_item("table_len", self.env.table_len)?;
        dict.set_item("led_suit", self.env.led_suit)?;
        dict.set_item("round_cards", self.env.round_cards)?;
        dict.set_item("current_trick", self.env.current_trick)?;
        dict.set_item("start_bidder", self.env.start_bidder)?;
        dict.set_item("bids_made", self.env.bids_made)?;
        dict.set_item("current_player", self.env.current_player)?;
        dict.set_item("leader", self.env.get_leader())?;
        dict.set_item(
            "phase",
            match self.env.phase {
                Phase::Bidding => "Bidding",
                Phase::Playing => "Playing",
                Phase::Terminal => "Terminal",
            },
        )?;
        dict.set_item("history_len", self.env.history_len)?;
        let history: Vec<(u8, u8, u8)> =
            self.env.history_events[..self.env.history_len as usize].to_vec();
        dict.set_item("history", history)?;
        Ok(dict.into())
    }

    #[getter]
    fn round_cards(&self) -> u8 {
        self.env.round_cards
    }

    #[getter]
    fn current_player(&self) -> u8 {
        self.env.current_player
    }

    #[getter]
    fn phase(&self) -> &'static str {
        match self.env.phase {
            Phase::Bidding => "Bidding",
            Phase::Playing => "Playing",
            Phase::Terminal => "Terminal",
        }
    }

    #[getter]
    fn bids(&self) -> Vec<i8> {
        self.env.bids.to_vec()
    }

    #[getter]
    fn tricks_won(&self) -> Vec<u8> {
        self.env.tricks_won.to_vec()
    }

    #[getter]
    fn led_suit(&self) -> u8 {
        self.env.led_suit
    }

    #[getter]
    fn table_len(&self) -> u8 {
        self.env.table_len
    }

    #[getter]
    fn start_bidder(&self) -> u8 {
        self.env.start_bidder
    }
}

/// Parallel vectorised training environment.
#[pyclass(module = "plump_engine")]
pub struct PyRustVecEnv {
    inner: RustVecEnv,
}

#[pymethods]
impl PyRustVecEnv {
    #[new]
    #[pyo3(signature = (num_envs = 256, seed = None))]
    fn new(num_envs: usize, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        });
        Self {
            inner: RustVecEnv::new(num_envs, seed),
        }
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    #[getter]
    fn rounds_completed(&self) -> u64 {
        self.inner.rounds_completed
    }

    /// `(toks [N,96] int64, feats [N,296] float32, legal [N,52] uint8,
    ///   is_bid [N] bool, players [N] uint8)`
    fn observe_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Py<PyArray2<i64>>,
        Py<PyArray2<f32>>,
        Py<PyArray2<u8>>,
        Py<PyArray1<bool>>,
        Py<PyArray1<u8>>,
    )> {
        let obs = self.inner.observe_batch();
        let n = obs.num_envs;
        Ok((
            flat_i64(py, obs.toks, n, MAX_SEQ)?.unbind(),
            flat_f32(py, obs.feats, n, FEAT_DIM)?.unbind(),
            flat_u8(py, obs.legal_masks, n, 52)?.unbind(),
            obs.is_bid_phase.into_pyarray_bound(py).unbind(),
            obs.current_players.into_pyarray_bound(py).unbind(),
        ))
    }

    /// Advance every environment. Returns the completed-round transitions and
    /// summaries as a dictionary of contiguous NumPy arrays.
    fn step_batch(&mut self, py: Python<'_>, actions: Vec<u8>) -> PyResult<PyObject> {
        if actions.len() != self.inner.num_envs() {
            return Err(PyValueError::new_err(format!(
                "expected {} actions, got {}",
                self.inner.num_envs(),
                actions.len()
            )));
        }
        let out = self.inner.step_batch(&actions);
        let m = out.transitions.len();

        let dict = PyDict::new_bound(py);
        let mut toks = Vec::with_capacity(m * MAX_SEQ);
        let mut feats = Vec::with_capacity(m * FEAT_DIM);
        let mut opp = Vec::with_capacity(m * 156);
        let mut acts = Vec::with_capacity(m);
        let mut rets = Vec::with_capacity(m);
        let mut final_tricks = Vec::with_capacity(m);
        let mut players = Vec::with_capacity(m);
        let mut is_play = Vec::with_capacity(m);
        let mut env_indices = Vec::with_capacity(m);
        for t in &out.transitions {
            toks.extend_from_slice(&t.toks);
            feats.extend_from_slice(&t.feats);
            opp.extend_from_slice(&t.opp_hands);
            acts.push(t.action);
            rets.push(t.ret);
            final_tricks.push(t.final_tricks as i64);
            players.push(t.player);
            is_play.push(t.is_play);
            env_indices.push(t.env_idx as i64);
        }
        dict.set_item("toks", flat_i64(py, toks, m, MAX_SEQ)?)?;
        dict.set_item("feats", flat_f32(py, feats, m, FEAT_DIM)?)?;
        dict.set_item("opp_hands", flat_f32(py, opp, m, 156)?)?;
        dict.set_item("actions", acts.into_pyarray_bound(py))?;
        dict.set_item("returns", rets.into_pyarray_bound(py))?;
        dict.set_item("final_tricks", final_tricks.into_pyarray_bound(py))?;
        dict.set_item("players", players.into_pyarray_bound(py))?;
        dict.set_item("is_play", is_play.into_pyarray_bound(py))?;
        dict.set_item("env_indices", env_indices.into_pyarray_bound(py))?;

        let s = out.summaries.len();
        let mut s_env = Vec::with_capacity(s);
        let mut s_cards = Vec::with_capacity(s);
        let mut s_start = Vec::with_capacity(s);
        let mut s_bids = Vec::with_capacity(s * 4);
        let mut s_tricks = Vec::with_capacity(s * 4);
        let mut s_points = Vec::with_capacity(s * 4);
        let mut s_rewards = Vec::with_capacity(s * 4);
        for summary in &out.summaries {
            s_env.push(summary.env_idx as i64);
            s_cards.push(summary.round_cards);
            s_start.push(summary.start_bidder);
            s_bids.extend(summary.bids.iter().map(|&b| b as i64));
            s_tricks.extend_from_slice(&summary.tricks_won);
            s_points.extend(summary.points.iter().map(|&v| v as i64));
            s_rewards.extend_from_slice(&summary.rewards);
        }
        let summaries = PyDict::new_bound(py);
        summaries.set_item("env_indices", s_env.into_pyarray_bound(py))?;
        summaries.set_item("round_cards", s_cards.into_pyarray_bound(py))?;
        summaries.set_item("start_bidder", s_start.into_pyarray_bound(py))?;
        summaries.set_item("bids", flat_i64(py, s_bids, s, 4)?)?;
        summaries.set_item("tricks_won", flat_u8(py, s_tricks, s, 4)?)?;
        summaries.set_item("points", flat_i64(py, s_points, s, 4)?)?;
        summaries.set_item("rewards", flat_f32(py, s_rewards, s, 4)?)?;
        dict.set_item("summaries", summaries)?;

        Ok(dict.into())
    }
}

/// Apply independent random suit permutations to a sampled batch.
///
/// Returns `(toks, feats, opp_hands, actions)`.
#[pyfunction]
fn augment_batch_numpy<'py>(
    py: Python<'py>,
    toks: PyReadonlyArray2<'py, i64>,
    feats: PyReadonlyArray2<'py, f32>,
    opp_hands: PyReadonlyArray2<'py, f32>,
    actions: PyReadonlyArray1<'py, i64>,
    is_play: PyReadonlyArray1<'py, bool>,
) -> PyResult<(
    Py<PyArray2<i64>>,
    Py<PyArray2<f32>>,
    Py<PyArray2<f32>>,
    Py<PyArray1<i64>>,
)> {
    let toks_shape = toks.shape();
    let feats_shape = feats.shape();
    let opp_shape = opp_hands.shape();
    let n = actions.shape()[0];
    if toks_shape != [n, MAX_SEQ]
        || feats_shape != [n, FEAT_DIM]
        || opp_shape != [n, 156]
        || is_play.shape()[0] != n
    {
        return Err(PyValueError::new_err(format!(
            "augment_batch_numpy shape mismatch: toks {:?}, feats {:?}, opp {:?}, actions {}, is_play {}",
            toks_shape,
            feats_shape,
            opp_shape,
            n,
            is_play.shape()[0]
        )));
    }

    let mut toks_vec: Vec<i64> = toks.as_array().iter().copied().collect();
    let mut feats_vec: Vec<f32> = feats.as_array().iter().copied().collect();
    let mut opp_vec: Vec<f32> = opp_hands.as_array().iter().copied().collect();
    let mut actions_vec: Vec<i64> = actions.as_array().iter().copied().collect();
    let is_play_vec: Vec<bool> = is_play.as_array().iter().copied().collect();

    let mut rng = StdRng::from_entropy();
    augment_batch(
        &mut toks_vec,
        &mut feats_vec,
        &mut opp_vec,
        &mut actions_vec,
        &is_play_vec,
        &mut rng,
    );

    Ok((
        flat_i64(py, toks_vec, n, MAX_SEQ)?.unbind(),
        flat_f32(py, feats_vec, n, FEAT_DIM)?.unbind(),
        flat_f32(py, opp_vec, n, 156)?.unbind(),
        actions_vec.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "terminal_reward")]
fn terminal_reward_py(tricks_won: u8, bid: i8) -> f32 {
    terminal_reward(tricks_won, bid)
}

#[pyfunction]
#[pyo3(name = "display_points")]
fn display_points_py(tricks_won: u8, bid: i8) -> i32 {
    display_points(tricks_won, bid)
}

#[pymodule]
fn plump_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFastEnv>()?;
    m.add_class::<PyRustVecEnv>()?;
    m.add_function(wrap_pyfunction!(augment_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(terminal_reward_py, m)?)?;
    m.add_function(wrap_pyfunction!(display_points_py, m)?)?;
    Ok(())
}
