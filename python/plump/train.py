# PPO trainer (§4).
#
# Rolls out `B` games in lockstep through the Rust engine (Phase 3 bridge),
# stores the fixed-shape §7.3 trajectory buffers, and applies the §4.3 PPO
# objective: disjoint bid/play heads, categorical value with trick-count
# factorization (§4.2), belief CE over ground-truth targets (§5.3), per-seat
# terminal MC returns with normalized advantages (§4.1), approx-KL early
# stopping, and an entropy floor.
#
# Run a short training loop + duplicate-deal eval with:
#   python -m plump.train --iters 5 --batch 512 --players 4 --cards 5

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .heuristic import play_heuristic
from .net import PlumpNet
from .rollout import RolloutDriver

EPS = 1e-8
_PLAYED = 6  # belief class: card already played (§5.3)
_UNDEALT = 7  # belief class: card in the dead stock


@dataclass
class PolicyLosses:
    policy: torch.Tensor
    value_ce: torch.Tensor
    value_consistency: torch.Tensor
    trick_ce: torch.Tensor
    belief_ce: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor
    approx_kl: torch.Tensor


def _select_logits(net_outputs: dict, is_bid: torch.Tensor) -> torch.Tensor:
    """Merge the disjoint bid/play heads into a [B, 52] action logits tensor.

    Bid ids live in 0..=10 and play ids are card indices 0..51. Bid rows are
    gathered from `bid` into the leading 11 slots; play rows pass through
    unchanged. Cross-phase actions are structurally impossible (§5.1).
    """
    bid = net_outputs["bid"]  # [B, 11]
    play = net_outputs["play"]  # [B, 52]
    logits = play.clone()
    bid_b = is_bid.nonzero(as_tuple=True)[0]
    if bid_b.numel():
        logits[bid_b, :11] = bid[bid_b]
    return logits


def _masked_logp(logits: torch.Tensor, action: torch.Tensor,
                 legal: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~legal, -1e30)
    return torch.log_softmax(masked, -1).gather(-1, action.unsqueeze(-1)).squeeze(-1)


def _approx_kl(new_logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    ratio = (new_logp - old_logp).exp()
    return (ratio - 1 - (new_logp - old_logp)).mean()


def compute_policy_loss(
    net: PlumpNet,
    net_outputs: dict,
    is_bid: torch.Tensor,
    action: torch.Tensor,
    legal: torch.Tensor,
    old_logp: torch.Tensor,
    ret: torch.Tensor,
    atom_idx: torch.Tensor,
    trick_tgt: torch.Tensor,
    belief_tgt: torch.Tensor,
    post_bid: torch.Tensor,
    bid: torch.Tensor,
    cfg: config.PpoConfig,
    entropy_coef: float,
) -> PolicyLosses:
    """PPO objective from §4.3 with all auxiliary losses gradient-scaled."""
    logits = _select_logits(net_outputs, is_bid)
    new_logp = _masked_logp(logits, action, legal)
    ratio = (new_logp - old_logp).exp()
    adv = ret - net.value(net_outputs["value"])
    adv = (adv - adv.mean()) / (adv.std() + EPS)

    policy_loss = -torch.min(
        ratio * adv,
        ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv,
    ).mean()

    value_ce = F.cross_entropy(net_outputs["value"], atom_idx)
    trick_ce = F.cross_entropy(net_outputs["tricks"], trick_tgt)

    belief_ce = _belief_ce(net, net_outputs["belief"], belief_tgt)

    # Post-bid factored value consistency: ||V - sum_t p(t)*score(bid,t)||² (§4.2).
    value_consistency = torch.tensor(0.0, device=adv.device)
    if post_bid.any():
        pb = post_bid
        p_tricks = torch.softmax(net_outputs["tricks"][pb], -1)  # [n, 11]
        v_factored = (p_tricks * net.score_table[bid[pb]]).sum(-1)
        value_consistency = F.mse_loss(net.value(net_outputs["value"][pb]), v_factored)

    ent = net.entropy(logits, legal).mean().clamp(min=cfg.entropy_floor)

    total = (
        policy_loss
        + cfg.value_coef * value_ce
        + cfg.consistency_coef * value_consistency
        + cfg.trick_coef * trick_ce
        + cfg.belief_coef * belief_ce
        - entropy_coef * ent
    )

    return PolicyLosses(
        policy=policy_loss,
        value_ce=value_ce,
        value_consistency=value_consistency,
        trick_ce=trick_ce,
        belief_ce=belief_ce,
        entropy=ent,
        total=total,
        approx_kl=_approx_kl(new_logp, old_logp),
    )


def _belief_ce(net: PlumpNet, belief_logits: torch.Tensor,
               belief_tgt: torch.Tensor) -> torch.Tensor:
    b, _ = belief_logits.shape
    logits = belief_logits.view(b, 52, net.n_belief_classes)
    return F.cross_entropy(logits.permute(0, 2, 1), belief_tgt)


def _gumbel_sample(logits: torch.Tensor) -> torch.Tensor:
    """Exact categorical sampling without `torch.multinomial` (gaps on MPS)."""
    u = torch.rand(logits.shape, device=logits.device).clamp(EPS, 1.0)
    g = -torch.log(-torch.log(u))
    return (logits + g).argmax(-1)


def sample_actions(net: PlumpNet, obs: torch.Tensor, legal: torch.Tensor,
                   is_bid: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples an action and logp from the current policy under the legal mask."""
    with torch.no_grad():
        out = net(obs)
        logits = _select_logits(out, is_bid).masked_fill(~legal, -1e30)
        action = _gumbel_sample(logits)
        logp = _masked_logp(logits, action, legal)
    return action, logp


class PPOTrainer:
    """Stages the fixed-shape §7.3 buffers and the §4 update loop."""

    def __init__(self, net: PlumpNet, env_cfg: config.EnvConfig,
                 ppo_cfg: config.PpoConfig, device: str):
        self.net = net.to(device)
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg
        self.device = device
        self.opt = torch.optim.Adam(net.parameters(), lr=ppo_cfg.lr)
        self.rollout_len = (1 + env_cfg.n_cards) * env_cfg.n_players
        self.iteration = 0
        self.last_kl = 0.0
        self.last_entropy = 0.0
        self.kl_early_stop = False  # update() cut short by approx-KL
        self.entropy_history: list[float] = []
        self.drv = RolloutDriver(
            env_cfg.n_players, env_cfg.n_cards, env_cfg.batch, seed=env_cfg.seed
        )
        # atom index lookup: round score value -> column in the value head.
        real = [a for a in net.score_atoms if a < 1_000]
        self._max_atom = max(real)
        lut = torch.zeros(self._max_atom + 1, dtype=torch.long)
        for i, a in enumerate(net.score_atoms):
            if a < 1_000:
                lut[a] = i
        self._atom_lut = lut
        self._alloc_buffers()

    def _alloc_buffers(self):
        # §7.3: pre-allocated once, fixed shapes, reused across iterations.
        D, B = self.rollout_len, self.env_cfg.batch
        self.obs_idx = torch.zeros(D, B, config.K_MAX, dtype=torch.int16)
        self.action = torch.zeros(D, B, dtype=torch.uint8)
        self.old_logp = torch.zeros(D, B, dtype=torch.float32)
        self.value = torch.zeros(D, B, dtype=torch.float32)
        self.legal = torch.zeros(D, B, 52, dtype=torch.bool)
        self.is_bid = torch.zeros(D, B, dtype=torch.bool)
        self.seat = torch.zeros(D, B, dtype=torch.uint8)
        self.ret = torch.zeros(D, B, dtype=torch.float32)
        self.atom_idx = torch.zeros(D, B, dtype=torch.long)
        self.trick_tgt = torch.zeros(D, B, dtype=torch.uint8)
        self.post_bid = torch.zeros(D, B, dtype=torch.bool)
        self.bid = torch.zeros(D, B, dtype=torch.uint8)
        self.belief_tgt = torch.zeros(D, B, 52, dtype=torch.uint8)

    def rollout(self):
        """One iteration: deal -> D policy steps -> per-seat MC returns (§4.1)."""
        drv = self.drv
        net = self.net
        D, B = self.rollout_len, self.env_cfg.batch
        drv.reset(self.env_cfg.seed + self.iteration)
        t0 = time.perf_counter()
        ent_acc = 0.0
        for t in range(D):
            idx, _ = drv.obs()
            obs = torch.from_numpy(idx).to(self.device, dtype=torch.int64)
            legal = torch.from_numpy(drv.legal_bool()).to(self.device)
            is_bid = torch.from_numpy(drv.is_bid != 0).to(self.device)
            with torch.no_grad():
                out = net(obs)
                logits = _select_logits(out, is_bid).masked_fill(~legal, -1e30)
                action = _gumbel_sample(logits)
                logp = _masked_logp(logits, action, legal)
                value = net.value(out["value"])
            ent_acc += net.entropy(logits, legal).mean().item()
            self.obs_idx[t] = torch.from_numpy(idx)
            self.action[t] = action.to("cpu")
            self.old_logp[t] = logp.to("cpu")
            self.value[t] = value.to("cpu")
            self.legal[t] = legal.to("cpu")
            self.is_bid[t] = is_bid.to("cpu")
            self.seat[t] = torch.from_numpy(drv.actor)
            self.belief_tgt[t] = torch.from_numpy(drv.belief_targets())
            drv.step(action.cpu().numpy().astype(np.uint8))
            drv.snapshot()

        # Terminal per-seat returns (§4.1): R_p = score_p, broadcast to that
        # seat's own decisions; no discounting, no cross-seat mixing.
        scores = np.zeros((B, self.env_cfg.n_players), dtype=np.float32)
        tricks = np.zeros((B, self.env_cfg.n_players), dtype=np.float32)
        for g in range(B):
            scores[g] = drv.round_scores(g)
            tricks[g] = drv.tricks(g)
        seats = self.seat.numpy()
        g_idx = np.arange(B)
        ret = torch.from_numpy(scores[g_idx[None, :], seats])  # [D, B]
        trick_tgt = torch.from_numpy(tricks[g_idx[None, :], seats])
        self.ret = ret
        self.trick_tgt = trick_tgt.to(torch.uint8)
        self.atom_idx = self._atom_lut[ret.long().clamp(0, self._max_atom)]
        # Each seat bids once at rows 0..P-1 (actor == row); recover every
        # row's own bid so the §4.2 factored value can use the real bid.
        g_idx = torch.arange(B)
        self.bid = self.action[: self.env_cfg.n_players][
            self.seat.long().clamp(max=self.env_cfg.n_players - 1), g_idx[None, :]
        ]
        self.post_bid = ~self.is_bid
        self.iteration += 1
        self.last_entropy = ent_acc / D
        self.entropy_history.append(self.last_entropy)
        return time.perf_counter() - t0

    def update(self):
        """§4.3: epochs x minibatches, index-permutation shuffling, KL stop."""
        cfg = self.ppo_cfg
        D, B = self.rollout_len, self.env_cfg.batch
        flat = D * B
        obs_f = self.obs_idx.reshape(flat, -1)
        action_f = self.action.reshape(flat)
        legal_f = self.legal.reshape(flat, 52)
        is_bid_f = self.is_bid.reshape(flat)
        old_logp_f = self.old_logp.reshape(flat)
        ret_f = self.ret.reshape(flat)
        atom_f = self.atom_idx.reshape(flat)
        trick_f = self.trick_tgt.reshape(flat)
        belief_f = self.belief_tgt.reshape(flat, 52)
        post_f = self.post_bid.reshape(flat)
        bid_f = self.bid.reshape(flat)
        mb = max(1, flat // cfg.minibatches)
        perm = torch.randperm(flat)
        d = self.device
        total_kl = 0.0
        n_updates = 0
        expected = cfg.epochs * cfg.minibatches
        for epoch in range(cfg.epochs):
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                obs = obs_f[idx].to(d, dtype=torch.int64)
                action = action_f[idx].to(d, dtype=torch.long)
                legal = legal_f[idx].to(d)
                is_bid = is_bid_f[idx].to(d)
                old_logp = old_logp_f[idx].to(d)
                ret = ret_f[idx].to(d)
                atom = atom_f[idx].to(d)
                trick = trick_f[idx].to(d)
                belief = belief_f[idx].to(d, dtype=torch.long)
                post = post_f[idx].to(d)
                bid = bid_f[idx].to(d, dtype=torch.long)
                out = self.net(obs)
                loss = compute_policy_loss(
                    self.net, out, is_bid, action, legal, old_logp, ret,
                    atom, trick, belief, post, bid, cfg, cfg.entropy_coef,
                )
                self.opt.zero_grad()
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                self.opt.step()
                kl = float(loss.approx_kl.item())
                total_kl += kl
                n_updates += 1
            if total_kl / n_updates > cfg.target_kl:
                break  # approx-KL early stopping (§4.3)
        self.kl_early_stop = n_updates < expected
        self.last_kl = total_kl / max(1, n_updates)
        return self.last_kl

    def save_checkpoint(self, path: Path | str):
        """Crash-safe snapshot: net + opt + iteration (buffers are re-derived)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "opt": self.opt.state_dict(),
                "iteration": self.iteration,
            },
            path,
        )

    def load_checkpoint(self, path: Path | str):
        """Resumes from a `save_checkpoint` snapshot."""
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ck["net"])
        self.opt.load_state_dict(ck["opt"])
        self.iteration = int(ck["iteration"])
        print(f"resumed from {path} (iteration {self.iteration})")

    def train(self, iterations: int, eval_every: int = 0,
              stall_limit: int = 3, save_dir: Path | str = "checkpoints",
              eval_games: int = 2048):
        save_dir = Path(save_dir)
        best_delta: float | None = None
        stalled = 0
        for it in range(iterations):
            t_roll = self.rollout()
            t_upd = time.perf_counter()
            kl = self.update()
            dt = time.perf_counter() - t_upd
            flags = []
            if self.kl_early_stop:
                flags.append("KL-STOP")
            if self.last_entropy < self.ppo_cfg.entropy_floor + 1e-3:
                flags.append("ENTROPY-COLLAPSED")
            print(
                f"iter {self.iteration - 1}: rollout {t_roll * 1e3:.1f} ms, "
                f"update {dt * 1e3:.1f} ms, approx-KL {kl:.4f}, "
                f"entropy {self.last_entropy:.4f}{(' [' + ', '.join(flags) + ']') if flags else ''}"
            )
            self.save_checkpoint(save_dir / "latest.pt")
            if eval_every > 0 and self.iteration % eval_every == 0:
                ev = self.evaluate(eval_games, seed=self.env_cfg.seed, baseline="random")
                delta = ev["delta"]
                improved = best_delta is None or delta > best_delta + 0.1
                if improved:
                    best_delta = delta
                    stalled = 0
                    self.save_checkpoint(save_dir / "best.pt")
                else:
                    stalled += 1
                print(
                    f"  eval[{self.iteration}]: net {ev['net']:.2f} vs random "
                    f"{ev['baseline']:.2f} pts/round (delta {delta:+.2f}, "
                    f"best {best_delta:+.2f})"
                )
                if stalled >= stall_limit:
                    print(f"  stopping: net-vs-random delta flat for {stalled} "
                          f"consecutive evals (best {best_delta:+.2f})")
                    break
            yield self.iteration - 1

    def evaluate(self, n_games: int, seed: int = 0, baseline: str = "random") -> dict:
        """Duplicate-deal eval (§9.1): the net and the baseline play identical
        deals; report the mean per-seat round score of each."""
        net_avg = _play_policy(self.net, self.env_cfg.n_players, self.env_cfg.n_cards,
                               n_games, seed, self.device)
        if baseline == "random":
            base_avg = _play_random(self.env_cfg.n_players, self.env_cfg.n_cards,
                                    n_games, seed)
        elif baseline == "heuristic":
            base_avg = play_heuristic(self.env_cfg.n_players, self.env_cfg.n_cards,
                                      n_games, seed)
        else:
            raise ValueError(baseline)
        return {"net": net_avg, "baseline": base_avg,
                "delta": net_avg - base_avg}


# -- duplicate-deal evaluators -------------------------------------------------

def _play_policy(net: PlumpNet, n_players: int, n_cards: int, n_games: int,
                 seed: int, device: str) -> float:
    drv = RolloutDriver(n_players, n_cards, n_games, seed=seed)
    for _ in range(drv.decisions_per_round):
        idx, _ = drv.obs()
        obs = torch.from_numpy(idx).to(device, dtype=torch.int64)
        legal = torch.from_numpy(drv.legal_bool()).to(device)
        is_bid = torch.from_numpy(drv.is_bid != 0).to(device)
        action, _ = sample_actions(net, obs, legal, is_bid, device)
        drv.step(action.cpu().numpy().astype(np.uint8))
        drv.snapshot()
    return np.mean([drv.round_scores(g) for g in range(n_games)])


def _play_random(n_players: int, n_cards: int, n_games: int, seed: int) -> float:
    drv = RolloutDriver(n_players, n_cards, n_games, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(drv.decisions_per_round):
        drv.play_random(rng)
    return np.mean([drv.round_scores(g) for g in range(n_games)])


def main(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--cards", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=config.default_device())
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--resume", action="store_true",
                    help="resume from --save-dir/latest.pt")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="eval vs random every N iters (0 = only at the end)")
    ap.add_argument("--stall", type=int, default=3,
                    help="stop after this many evals without net-vs-random progress")
    args = ap.parse_args(argv)

    env_cfg = config.EnvConfig(
        n_players=args.players, n_cards=args.cards, batch=args.batch, seed=args.seed
    )
    net = PlumpNet()
    trainer = PPOTrainer(net, env_cfg, config.PpoConfig(), args.device)
    if args.resume:
        trainer.load_checkpoint(Path(args.save_dir) / "latest.pt")
    t0 = time.perf_counter()
    for _ in trainer.train(args.iters, eval_every=args.eval_every,
                           stall_limit=args.stall, save_dir=args.save_dir,
                           eval_games=args.eval_games):
        pass
    print(f"training took {time.perf_counter() - t0:.1f}s")
    trainer.save_checkpoint(Path(args.save_dir) / "latest.pt")

    ev = trainer.evaluate(args.eval_games, seed=args.seed, baseline="random")
    print(f"eval vs random: net {ev['net']:.2f} vs random {ev['baseline']:.2f}"
          f" pts/round (delta {ev['delta']:+.2f})")
    ev = trainer.evaluate(args.eval_games, seed=args.seed, baseline="heuristic")
    print(f"eval vs heuristic: net {ev['net']:.2f} vs heuristic {ev['baseline']:.2f}"
          f" pts/round (delta {ev['delta']:+.2f})")


if __name__ == "__main__":
    main()
