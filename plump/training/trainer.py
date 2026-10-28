"""Self-play trainer: DMC actor-critic with auxiliary belief/tricks heads."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import plump_engine

from ..agents import epsilon_greedy, heuristic_bid, heuristic_play, masked_argmax
from ..checkpoint import load_checkpoint, save_checkpoint
from ..config import BID_BASE, PlumpTransformerConfig, TrainConfig
from ..models.transformer import PlumpTransformer
from .buffer import RingBuffer
from .league import CHAMPION_BASE, FINAL_BOSS, REGULAR, ROOKIE, TACTICIAN, League

COMBO_MASKS = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))


def format_gauntlet_ladder(results: Dict[str, Dict[str, float]], round_num: int) -> str:
    boss_info = [
        ("Rookie", "Lv.1 Rookie"),
        ("Regular", "Lv.2 Regular"),
        ("Tactician", "Lv.3 Tactician"),
        ("Final Boss", "Lv.4 Final Boss"),
    ]
    statuses = []
    fighting_found = False
    for name, _ in boss_info:
        data = results.get(name, {})
        margin = data.get("margin", 0.0)
        if not fighting_found:
            if margin > 0.0:
                statuses.append("[✓ DEFEATED]")
            else:
                statuses.append("[⚔ FIGHTING]")
                fighting_found = True
        else:
            statuses.append("[🔒 LOCKED  ]")

    title = f" BOSS GAUNTLET (Round {round_num:,}) "
    inner_width = 78
    dashes = max(0, inner_width - len(title))
    left_dashes = dashes // 2
    right_dashes = dashes - left_dashes
    top = "╭" + "─" * left_dashes + title + "─" * right_dashes + "╮"
    bot = "╰" + "─" * inner_width + "╯"

    lines = [top]
    for (name, label), status in zip(boss_info, statuses):
        data = results.get(name, {})
        ai_pts = data.get("ai_avg_points", 0.0)
        bot_pts = data.get("bot_avg_points", 0.0)
        margin = data.get("margin", 0.0)
        content = f"  {status} {label:<16} AI: {ai_pts:5.2f} pts | Bot: {bot_pts:5.2f} pts | Margin: {margin:+5.2f}"
        pad = inner_width - len(content)
        lines.append("│" + content + " " * max(0, pad) + "│")
    lines.append(bot)
    return "\n".join(lines)


class PlumpTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model_cfg: Optional[PlumpTransformerConfig] = None,
        resume: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.amp_enabled = cfg.use_amp and self.device.type == "cuda"
        self.rng_np = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        payload = load_checkpoint(resume, self.device) if resume else None
        if payload and "model" in payload:
            model_cfg = PlumpTransformerConfig.from_dict(payload.get("config", {}))
        self.model = PlumpTransformer(model_cfg or PlumpTransformerConfig())
        self.model.to(self.device)
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        if payload:
            state = payload["model"] if "model" in payload else payload
            self.model.load_state_dict(state)
            if "optimizer" in payload:
                try:
                    self.optimizer.load_state_dict(payload["optimizer"])
                except (ValueError, KeyError):
                    pass

        self.vec_env = plump_engine.PyRustVecEnv(cfg.num_envs, seed=cfg.seed)
        self.play_buffer = RingBuffer(cfg.replay_play)
        self.bid_buffer = RingBuffer(cfg.replay_bid)
        self.league = League(self.device, cfg.max_champions, seed=cfg.seed)

        self.rounds_seen = int(payload.get("rounds_seen", 0)) if payload else 0
        self.grad_steps = int(payload.get("grad_steps", 0)) if payload else 0
        self.best_score = float(payload.get("best_score", -math.inf)) if payload else -math.inf
        self.last_eval_metrics: Optional[Dict] = None
        self.last_gauntlet: Optional[Dict[str, Dict[str, float]]] = None
        self.last_train_metrics: Dict[str, float] = {}
        self.env_steps = 0
        self.train_calls = 0
        if payload and payload.get("league"):
            self.league.load_champions(
                payload["league"], payload.get("league_scores")
            )

        self.learner_masks = np.ones((cfg.num_envs, 4), dtype=np.bool_)
        self.opp_policy = np.full(cfg.num_envs, -1, dtype=np.int64)
        self._init_assignments()
        self._start_time = time.time()
        self._start_rounds = self.rounds_seen

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _init_assignments(self) -> None:
        n_league = int(self.cfg.num_envs * self.cfg.league_fraction)
        for e in range(n_league):
            combo = COMBO_MASKS[e % len(COMBO_MASKS)]
            self.learner_masks[e] = False
            self.learner_masks[e, list(combo)] = True
            self.opp_policy[e] = self.league.sample_policy()

    def _rotate_assignment(self, env: int) -> None:
        if self.opp_policy[env] < 0:
            return
        current = tuple(np.flatnonzero(self.learner_masks[env]).tolist())
        index = COMBO_MASKS.index(current) if current in COMBO_MASKS else 0
        combo = COMBO_MASKS[(index + 1) % len(COMBO_MASKS)]
        self.learner_masks[env] = False
        self.learner_masks[env, list(combo)] = True
        self.opp_policy[env] = self.league.sample_policy()

    def _model_actions(
        self,
        model: PlumpTransformer,
        toks: np.ndarray,
        feats: np.ndarray,
        legal: np.ndarray,
        indices: np.ndarray,
        bidding: bool,
        explore: bool,
    ) -> np.ndarray:
        phase = "bid" if bidding else "play"
        n_actions = 11 if bidding else 52
        tokens = torch.from_numpy(toks[indices]).long().to(self.device, non_blocking=True)
        features = torch.from_numpy(feats[indices]).float().to(self.device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.amp_enabled,
            ):
                q = model(tokens, features, phase=phase)
        q = q.float().cpu().numpy()
        legal_rows = legal[indices][:, :n_actions].astype(np.bool_)
        if not explore:
            return masked_argmax(q, legal_rows).astype(np.uint8)
        if bidding:
            eps = np.linspace(0.02, 0.20, self.cfg.num_envs)[indices]
        else:
            eps = np.linspace(0.005, 0.03, self.cfg.num_envs)[indices]
        return epsilon_greedy(q, legal_rows, eps, self.rng_np)

    def _random_actions(
        self, legal: np.ndarray, indices: np.ndarray, bidding: bool
    ) -> np.ndarray:
        n_actions = 11 if bidding else 52
        actions = np.zeros(len(indices), dtype=np.uint8)
        for j, i in enumerate(indices):
            options = np.flatnonzero(legal[i, :n_actions])
            if len(options):
                actions[j] = options[self.rng_np.integers(0, len(options))]
        return actions

    @staticmethod
    def _heuristic_actions(
        toks: np.ndarray,
        feats: np.ndarray,
        legal: np.ndarray,
        indices: np.ndarray,
        bidding: bool,
    ) -> np.ndarray:
        actions = np.zeros(len(indices), dtype=np.uint8)
        for j, i in enumerate(indices):
            if bidding:
                round_cards = int(toks[i, 1]) - BID_BASE
                actions[j] = heuristic_bid(legal[i], round_cards)
            else:
                bid = max(0, int(round(float(feats[i, 276]) * 10)))
                tricks = max(0, int(round(float(feats[i, 280]) * 10)))
                actions[j] = heuristic_play(legal[i], bid, tricks)
        return actions

    def _select_actions(self, obs) -> np.ndarray:
        toks, feats, legal, is_bid, players = obs
        n = len(toks)
        actions = np.zeros(n, dtype=np.uint8)
        controlled = np.ones(n, dtype=np.bool_)
        for e in range(n):
            if self.opp_policy[e] >= 0 and not self.learner_masks[e, int(players[e])]:
                controlled[e] = False

        for bidding in (True, False):
            phase_idx = np.flatnonzero(np.asarray(is_bid, dtype=np.bool_) == bidding)
            if len(phase_idx) == 0:
                continue
            learner_idx = phase_idx[controlled[phase_idx]]
            if len(learner_idx):
                actions[learner_idx] = self._model_actions(
                    self.model, toks, feats, legal, learner_idx, bidding, True
                )
            opponent_idx = phase_idx[~controlled[phase_idx]]
            if len(opponent_idx):
                policies = self.opp_policy[opponent_idx]
                # Fast vectorised Rust bots for policies 0..=3 (ROOKIE, REGULAR, TACTICIAN, FINAL_BOSS)
                bot_mask = (policies >= 0) & (policies < CHAMPION_BASE)
                if bot_mask.any():
                    bot_group = opponent_idx[bot_mask]
                    bot_levels = np.zeros(n, dtype=np.uint8)
                    bot_levels[bot_group] = policies[bot_mask].astype(np.uint8)
                    bot_acts = np.asarray(self.vec_env.bot_actions(bot_levels))
                    actions[bot_group] = bot_acts[bot_group]
                # Frozen champion models for policies >= CHAMPION_BASE
                champ_mask = policies >= CHAMPION_BASE
                if champ_mask.any():
                    for policy in np.unique(policies[champ_mask]):
                        group = opponent_idx[policies == policy]
                        champion = self.league.model_for(int(policy))
                        if champion is not None:
                            actions[group] = self._model_actions(
                                champion, toks, feats, legal, group, bidding, False
                            )
                        else:
                            bot_levels = np.full(n, TACTICIAN, dtype=np.uint8)
                            bot_acts = np.asarray(self.vec_env.bot_actions(bot_levels))
                            actions[group] = bot_acts[group]
        return actions

    def _ingest(self, out: dict) -> None:
        env_idx = np.asarray(out["env_indices"], dtype=np.int64)
        if len(env_idx) == 0:
            return
        players = np.asarray(out["players"], dtype=np.int64)
        keep = self.learner_masks[env_idx, players]
        if not keep.any():
            return

        def selected(name: str) -> np.ndarray:
            return np.asarray(out[name])[keep]

        is_play = selected("is_play").astype(np.bool_)
        for play_phase in (True, False):
            mask = is_play if play_phase else ~is_play
            if not mask.any():
                continue
            buffer = self.play_buffer if play_phase else self.bid_buffer
            buffer.add_batch(
                selected("toks")[mask],
                selected("feats")[mask],
                selected("actions")[mask],
                selected("returns")[mask],
                selected("opp_hands")[mask],
                selected("final_tricks")[mask],
                is_play[mask],
            )

    def train_step(self) -> Optional[Dict[str, float]]:
        batch = self.cfg.batch_size
        if len(self.play_buffer) < batch:
            return None

        self.train_calls += 1
        # Train bidder on every step as soon as buffer is ready.
        use_bid = len(self.bid_buffer) >= batch
        phases: list[tuple[str, RingBuffer]] = [("play", self.play_buffer)]
        if use_bid:
            phases.append(("bid", self.bid_buffer))

        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        metrics: Dict[str, float] = {}
        for phase, buffer in phases:
            sample = buffer.sample(batch, augment=True, rng=self.rng_np)
            tokens = torch.from_numpy(sample["toks"]).long().to(self.device, non_blocking=True)
            feats = torch.from_numpy(sample["feats"]).float().to(self.device, non_blocking=True)
            actions = torch.from_numpy(sample["actions"]).long().to(self.device, non_blocking=True)
            returns = torch.from_numpy(sample["returns"]).float().to(self.device, non_blocking=True)
            opp_hands = torch.from_numpy(sample["opp_hands"]).float().to(self.device, non_blocking=True)
            final_tricks = torch.from_numpy(sample["final_tricks"]).long().to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.amp_enabled,
            ):
                q, belief, tricks_logits = self.model.forward_with_aux(
                    tokens, feats, phase
                )
                q_taken = q.gather(1, actions.unsqueeze(1)).squeeze(1).float()
                q_loss = F.smooth_l1_loss(q_taken, returns)
                belief_loss = F.binary_cross_entropy_with_logits(
                    belief.float().reshape(-1, 156), opp_hands.float()
                )
                tricks_loss = F.cross_entropy(
                    tricks_logits.float(), final_tricks
                )
                loss = (
                    q_loss
                    + self.cfg.belief_loss_weight * belief_loss
                    + self.cfg.tricks_loss_weight * tricks_loss
                )
            total_loss = total_loss + loss
            metrics[f"{phase}_q"] = float(q_loss.detach())
            metrics[f"{phase}_belief"] = float(belief_loss.detach())
            metrics[f"{phase}_tricks"] = float(tricks_loss.detach())

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        )
        self.optimizer.step()
        self.grad_steps += 1
        metrics["loss"] = float(total_loss.detach())
        metrics["grad_norm"] = float(grad_norm)
        if "bid_q" in metrics:
            metrics["q_loss"] = 0.5 * (metrics["bid_q"] + metrics["play_q"])
            metrics["belief_loss"] = 0.5 * (
                metrics["bid_belief"] + metrics["play_belief"]
            )
        else:
            metrics["q_loss"] = metrics["play_q"]
            metrics["belief_loss"] = metrics["play_belief"]
        self.last_train_metrics = metrics
        return metrics

    def _eval_actions(self, obs, learner_seat: int, vec=None) -> np.ndarray:
        toks, feats, legal, is_bid, players = obs
        n = len(toks)
        actions = np.zeros(n, dtype=np.uint8)
        controlled = np.asarray(players) == learner_seat
        for bidding in (True, False):
            phase_idx = np.flatnonzero(np.asarray(is_bid, dtype=np.bool_) == bidding)
            if len(phase_idx) == 0:
                continue
            learner_idx = phase_idx[controlled[phase_idx]]
            if len(learner_idx):
                actions[learner_idx] = self._model_actions(
                    self.model, toks, feats, legal, learner_idx, bidding, False
                )
            opponent_idx = phase_idx[~controlled[phase_idx]]
            if len(opponent_idx):
                champion = self.league.champion()
                if champion is None:
                    if vec is not None:
                        bot_acts = np.asarray(
                            vec.bot_actions(np.full(n, TACTICIAN, dtype=np.uint8))
                        )
                        actions[opponent_idx] = bot_acts[opponent_idx]
                    else:
                        actions[opponent_idx] = self._heuristic_actions(
                            toks, feats, legal, opponent_idx, bidding
                        )
                else:
                    actions[opponent_idx] = self._model_actions(
                        champion, toks, feats, legal, opponent_idx, bidding, False
                    )
        return actions

    def evaluate(self) -> Dict:
        self.model.eval()
        target = self.cfg.eval_batch_rounds
        total_points = 0.0
        total_opp_points = 0.0
        total_rounds = 0
        total_made = 0
        total_tricks = 0
        per_seat = []
        for seat in range(4):
            vec = plump_engine.PyRustVecEnv(
                self.cfg.eval_envs, seed=self.cfg.seed + 777 + seat * 31
            )
            seat_rounds = 0
            seat_points = 0.0
            seat_opp_points = 0.0
            seat_made = 0
            seat_tricks = 0
            while seat_rounds < target:
                obs = vec.observe_batch()
                actions = self._eval_actions(obs, seat, vec=vec)
                out = vec.step_batch(actions.tolist())
                summaries = out["summaries"]
                for j in range(len(summaries["env_indices"])):
                    if seat_rounds >= target:
                        break
                    bid = int(summaries["bids"][j, seat])
                    tricks = int(summaries["tricks_won"][j, seat])
                    points = float(summaries["points"][j, seat])
                    opp_pts = float(
                        (summaries["points"][j].sum() - summaries["points"][j, seat])
                        / 3.0
                    )
                    seat_points += points
                    seat_opp_points += opp_pts
                    seat_tricks += tricks
                    seat_made += int(tricks == bid)
                    seat_rounds += 1
            per_seat.append(
                {
                    "seat": seat,
                    "rounds": seat_rounds,
                    "avg_points": seat_points / max(1, seat_rounds),
                    "opp_avg_points": seat_opp_points / max(1, seat_rounds),
                    "margin": (seat_points - seat_opp_points) / max(1, seat_rounds),
                    "made_rate": seat_made / max(1, seat_rounds),
                    "avg_tricks": seat_tricks / max(1, seat_rounds),
                }
            )
            total_points += seat_points
            total_opp_points += seat_opp_points
            total_rounds += seat_rounds
            total_made += seat_made
            total_tricks += seat_tricks
            del vec
        avg_score = total_points / max(1, total_rounds)
        opp_avg_score = total_opp_points / max(1, total_rounds)
        margin = avg_score - opp_avg_score
        metrics = {
            "rounds": total_rounds,
            "avg_score": avg_score,
            "opp_avg_score": opp_avg_score,
            "margin": margin,
            "made_rate": total_made / max(1, total_rounds),
            "avg_tricks": total_tricks / max(1, total_rounds),
            "per_seat": per_seat,
        }
        self.last_eval_metrics = metrics
        return metrics

    def _evaluate_against_bot(
        self, bot_level: int, rounds_per_seat: int = 20, num_envs: int = 20
    ) -> Dict[str, float]:
        self.model.eval()
        total_points = 0.0
        total_opp_points = 0.0
        total_rounds = 0
        total_made = 0
        total_tricks = 0

        # 20 envs = all 20 rounds of a real Plump game (10..1, 1..10)
        num_envs = 20

        for seat in range(4):
            vec = plump_engine.PyRustVecEnv(
                num_envs, seed=self.cfg.seed + 1000 + bot_level * 100 + seat * 17
            )
            finished_envs = set()
            while len(finished_envs) < num_envs:
                obs = vec.observe_batch()
                toks, feats, legal, is_bid, players = obs
                n = len(toks)
                actions = np.zeros(n, dtype=np.uint8)
                controlled = np.asarray(players) == seat

                for bidding in (True, False):
                    phase_idx = np.flatnonzero(np.asarray(is_bid, dtype=np.bool_) == bidding)
                    if len(phase_idx) == 0:
                        continue
                    learner_idx = phase_idx[controlled[phase_idx]]
                    if len(learner_idx):
                        actions[learner_idx] = self._model_actions(
                            self.model, toks, feats, legal, learner_idx, bidding, False
                        )

                opp_idx = np.flatnonzero(~controlled)
                if len(opp_idx):
                    bot_acts = np.asarray(
                        vec.bot_actions(np.full(n, bot_level, dtype=np.uint8))
                    )
                    actions[opp_idx] = bot_acts[opp_idx]

                out = vec.step_batch(actions.tolist())
                summaries = out["summaries"]
                for j, env_idx in enumerate(summaries["env_indices"]):
                    env_idx = int(env_idx)
                    # Only count the FIRST round finished by each of the 20 tables
                    if env_idx in finished_envs:
                        continue
                    finished_envs.add(env_idx)
                    bid = int(summaries["bids"][j, seat])
                    tricks = int(summaries["tricks_won"][j, seat])
                    points = float(summaries["points"][j, seat])
                    opp_pts = float(
                        (summaries["points"][j].sum() - summaries["points"][j, seat])
                        / 3.0
                    )
                    total_points += points
                    total_opp_points += opp_pts
                    total_tricks += tricks
                    total_made += int(tricks == bid)
                    total_rounds += 1
            del vec

        avg_score = total_points / max(1, total_rounds)
        opp_avg_score = total_opp_points / max(1, total_rounds)
        margin = avg_score - opp_avg_score
        made_rate = total_made / max(1, total_rounds)
        avg_tricks = total_tricks / max(1, total_rounds)

        return {
            "ai_avg_points": avg_score,
            "bot_avg_points": opp_avg_score,
            "margin": margin,
            "made_rate": made_rate,
            "avg_tricks": avg_tricks,
            "rounds": total_rounds,
        }

    def evaluate_gauntlet(
        self, rounds_per_seat: int = 8
    ) -> Dict[str, Dict[str, float]]:
        bosses = [
            ("Rookie", ROOKIE),
            ("Regular", REGULAR),
            ("Tactician", TACTICIAN),
            ("Final Boss", FINAL_BOSS),
        ]
        results = {}
        for name, level in bosses:
            results[name] = self._evaluate_against_bot(level, rounds_per_seat=rounds_per_seat)
        return results

    def save(self, best: bool = False) -> None:
        path = self.cfg.best_path if best else self.cfg.latest_path
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.model.cfg,
            rounds_seen=self.rounds_seen,
            grad_steps=self.grad_steps,
            best_score=self.best_score,
            eval_metrics=self.last_eval_metrics,
            league=self.league.state_dicts(),
            league_scores=list(self.league.scores),
        )

    def write_session(self) -> None:
        path = self.cfg.session_path
        path.parent.mkdir(parents=True, exist_ok=True)
        elapsed = max(1e-9, time.time() - self._start_time)
        metrics = self.last_eval_metrics or {}
        tm = self.last_train_metrics or {}
        lines = [
            "Plump Transformer training session",
            f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"device: {self.device}",
            f"rounds_seen: {self.rounds_seen}",
            f"grad_steps: {self.grad_steps}",
            f"best_score: {self.best_score:.4f}",
            f"rounds_per_second: "
            f"{(self.rounds_seen - self._start_rounds) / elapsed:.1f}",
            f"train_q_loss: {tm.get('q_loss', float('nan')):.4f}",
            f"train_belief_loss: {tm.get('belief_loss', float('nan')):.4f}",
            f"eval_avg_score: {metrics.get('avg_score', float('nan')):.4f}",
            f"eval_opp_avg_score: {metrics.get('opp_avg_score', float('nan')):.4f}",
            f"eval_margin: {metrics.get('margin', float('nan')):.4f}",
            f"eval_made_rate: {metrics.get('made_rate', float('nan')):.4f}",
            f"eval_avg_tricks: {metrics.get('avg_tricks', float('nan')):.4f}",
            f"league_champions: {len(self.league)}",
        ]
        for seat in metrics.get("per_seat", []):
            lines.append(
                "seat {seat}: score={avg_points:.3f} opp={opp_avg_points:.3f} margin={margin:+.3f} "
                "made={made_rate:.3f} tricks={avg_tricks:.2f}".format(**seat)
            )
        if self.last_gauntlet:
            lines.append("")
            lines.append(format_gauntlet_ladder(self.last_gauntlet, self.rounds_seen))
            for name, d in self.last_gauntlet.items():
                lines.append(
                    f"gauntlet_{name.lower().replace(' ', '_')}: ai={d['ai_avg_points']:.2f} "
                    f"bot={d['bot_avg_points']:.2f} margin={d['margin']:+.2f} made={d['made_rate']:.1%}"
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _evaluate_and_checkpoint(self) -> Dict:
        was_training = self.model.training
        metrics = self.evaluate()
        gauntlet = self.evaluate_gauntlet()
        self.last_gauntlet = gauntlet
        self.model.train(was_training)

        gauntlet_score = sum(d["ai_avg_points"] for d in gauntlet.values()) / len(gauntlet)
        gauntlet_margin = sum(d["margin"] for d in gauntlet.values()) / len(gauntlet)
        added = False
        if metrics["margin"] > 0.05 or len(self.league) == 0:
            added = bool(self.league.add_champion(self.model, gauntlet_score))

        improved = (gauntlet_score > self.best_score) and (gauntlet_margin > 0.0)
        if improved:
            self.best_score = gauntlet_score
            self.save(best=True)
        elif not self.cfg.best_path.exists():
            self.save(best=True)

        self.save(best=False)
        self.write_session()

        ladder_str = format_gauntlet_ladder(gauntlet, self.rounds_seen)
        print("\n" + ladder_str, flush=True)

        elapsed = max(1e-9, time.time() - self._start_time)
        tm = self.last_train_metrics or {}
        print(
            f"[eval] rounds={self.rounds_seen} score={metrics['avg_score']:.3f} "
            f"opp_score={metrics['opp_avg_score']:.3f} margin={metrics['margin']:+.2f} "
            f"made={metrics['made_rate']:.1%} tricks={metrics['avg_tricks']:.2f} "
            f"q={tm.get('q_loss', float('nan')):.3f} belief={tm.get('belief_loss', float('nan')):.3f} "
            f"league={len(self.league)} new_champ={'yes' if added else 'no'} "
            f"best={self.best_score:.3f} rps={(self.rounds_seen - self._start_rounds) / elapsed:.1f}",
            flush=True,
        )
        return metrics

    def _step(self) -> None:
        self.env_steps += 1
        obs = self.vec_env.observe_batch()
        actions = self._select_actions(obs)
        out = self.vec_env.step_batch(actions.tolist())
        summaries = out["summaries"]
        self._ingest(out)
        for env in summaries["env_indices"]:
            self._rotate_assignment(int(env))
        self.rounds_seen += len(summaries["env_indices"])

    def _log_progress(self, last_log: float) -> float:
        now = time.time()
        if now - last_log < 5.0:
            return last_log
        elapsed = max(1e-9, now - self._start_time)
        print(
            "[train] rounds={rounds} rounds/s={rps:.1f} grad_steps={steps} "
            "play_buf={play} bid_buf={bid} league={league}".format(
                rounds=self.rounds_seen,
                rps=(self.rounds_seen - self._start_rounds) / elapsed,
                steps=self.grad_steps,
                play=len(self.play_buffer),
                bid=len(self.bid_buffer),
                league=len(self.league),
            ),
            flush=True,
        )
        return now

    def run(self, max_rounds: int) -> None:
        last_eval = self.rounds_seen
        last_log = time.time()
        try:
            while self.rounds_seen < max_rounds:
                self._step()
                if self.env_steps % 2 == 0:
                    self.train_step()
                if self.rounds_seen - last_eval >= self.cfg.eval_every:
                    self._evaluate_and_checkpoint()
                    last_eval = self.rounds_seen
                last_log = self._log_progress(last_log)
        except KeyboardInterrupt:
            print("[train] interrupted, saving checkpoint...", flush=True)
            self.save(best=False)
            self.write_session()
            return
        self._evaluate_and_checkpoint()

    def run_test_run(self, rounds: int = 2000, updates: int = 10) -> None:
        start_rounds = self.rounds_seen
        start_steps = self.grad_steps
        last_log = time.time()
        iterations = 0
        while True:
            iterations += 1
            self._step()
            if self.env_steps % 2 == 0:
                self.train_step()
            last_log = self._log_progress(last_log)
            enough_rounds = self.rounds_seen - start_rounds >= rounds
            enough_updates = self.grad_steps - start_steps >= updates
            if enough_rounds and enough_updates:
                break
            if iterations > 5000:
                break
        self._evaluate_and_checkpoint()
        print(
            "[test-run] completed rounds={rounds} grad_steps={steps}".format(
                rounds=self.rounds_seen - start_rounds,
                steps=self.grad_steps - start_steps,
            ),
            flush=True,
        )
