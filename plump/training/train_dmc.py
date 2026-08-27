# plump/training/train_dmc.py — DMC + Transformer + league, restart-from-scratch.
# Composes plump/env + encode/tokenizer + models/transformer + league.
# Single-process for simplicity, but decoupled for future 24-actor batched.

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ..env import PlumpEnv
from ..encode.tokenizer import encode_decision, VOCAB, FEAT_DIM, PAD_TOK
from ..models.transformer import PlumpTransformer, PlumpTransformerConfig
from ..env.cards import CARD_INDEX, INDEX_TO_CARD
from ..encode.legacy import mask_actions
from ..config.settings import TRAIN, HOUSE, terminal_reward
from .rewards import shaped_reward
from .league import League


def play_one_game_transformer(env: PlumpEnv, model: PlumpTransformer, league: League, device, epsilon=0.1):
    """One game where seat 0 = learner (transformer), seats 1-3 = league samples.
    Returns episode data for DMC: returns are computed at round end.
    """
    env.new_round(random.randint(1, 10))
    # For DMC we collect per-decision tuples for learner only
    learner_bids: list[dict] = []
    learner_plays: list[dict] = []

    # Sample league opponents for this game
    opp_for_seat = {}
    sampled = league.sample_for_game()
    for seat, (name, opp) in zip([1, 2, 3], sampled):
        opp_for_seat[seat] = (name, opp)

    # Bidding phase — learner is seat 0 only
    for turn in range(4):
        is_learner = turn == 0
        if is_learner:
            toks, feats = encode_decision(env, turn)
            legal = mask_actions(env, turn, "bid")
            # transformer inference
            t_toks = torch.tensor(np.array([toks]), dtype=torch.long, device=device)
            t_feats = torch.tensor(np.array([feats]), dtype=torch.float32, device=device)
            model.eval()
            with torch.no_grad():
                q = model(t_toks, t_feats, phase="bid")[0]
            # epsilon-greedy over legal
            if random.random() < epsilon:
                cands = [i for i, v in enumerate(legal) if v]
                bid = random.choice(cands)
            else:
                q_masked = q.clone()
                q_masked[~torch.tensor(legal, device=device)] = -1e9
                bid = int(q_masked.argmax().item())
            learner_bids.append({"player": turn, "toks": toks, "feats": feats, "action": bid, "legal": legal})
            env.bids[turn] = bid
            env.record_bid(turn, bid)
        else:
            # league opponent
            _, opp = opp_for_seat[turn]
            bid = opp.act_bid(env, turn)
            env.bids[turn] = bid
            env.record_bid(turn, bid)

    # Play — learner may be leader or not, but we only learn from learner seat
    try:
        max_bid = max(env.bids)
        cur = next(i for i, b in enumerate(env.bids) if b == max_bid)
    except ValueError:
        cur = 0

    # Track shaped rewards for learner
    learner_play_shaped: list[float] = []

    for trick in range(env.round_cards):
        trick_learner_idx = None
        for _ in range(4):
            is_learner = cur == 0
            if is_learner:
                toks, feats = encode_decision(env, cur)
                legal = mask_actions(env, cur, "play")
                t_toks = torch.tensor(np.array([toks]), dtype=torch.long, device=device)
                t_feats = torch.tensor(np.array([feats]), dtype=torch.float32, device=device)
                with torch.no_grad():
                    q = model(t_toks, t_feats, phase="play")[0]
                if random.random() < epsilon:
                    cands = [CARD_INDEX[c] for c in env.legal_cards(cur)]
                    # map back to card
                    legal_cards = env.legal_cards(cur)
                    card = random.choice(legal_cards)
                    action = CARD_INDEX[card]
                else:
                    q_masked = q.clone()
                    mask = torch.tensor(legal, dtype=torch.bool, device=device)
                    q_masked[~mask] = -1e9
                    action = int(q_masked.argmax().item())
                    card = INDEX_TO_CARD[action]
                learner_plays.append({"player": cur, "toks": toks, "feats": feats, "action": action, "legal": legal, "card": card})
                trick_learner_idx = len(learner_plays) - 1
                env.play_card(cur, card)
            else:
                _, opp = opp_for_seat[cur] if cur in opp_for_seat else (None, None)
                # if cur not in sampled (when leader is learner, opp seats still 1,2,3 but cur may be 0)
                # fallback to random if needed
                if opp is None:
                    # shouldn't happen, but use random legal
                    card = random.choice(env.legal_cards(cur))
                else:
                    card = opp.act_play(env, cur)
                env.play_card(cur, card)
            cur = (cur + 1) % 4
        winner, _ = env.resolve_trick()
        # shaped for learner if they played this trick
        if trick_learner_idx is not None:
            # learner's tricks after resolve
            is_win = winner == 0
            r = shaped_reward(is_win, env.tricks_won[0], env.bids[0])
            # store shaped for this trick's decision
            learner_plays[trick_learner_idx]["shaped"] = r
        else:
            # learner didn't play this trick? impossible if 4 players and learner is one, they play every trick
            pass
        cur = winner

    # Compute DMC returns for learner: terminal + shaped per play, terminal only for bids
    # Bids: single terminal return
    bid_returns = []
    for d in learner_bids:
        r = terminal_reward(env.tricks_won[d["player"]], env.bids[d["player"]])
        bid_returns.append(r)

    # Plays: DMC sum of shaped + terminal at last play
    play_returns = []
    for i, d in enumerate(learner_plays):
        is_last = i == len(learner_plays) - 1
        shaped = d.get("shaped", 0.0)
        term = terminal_reward(env.tricks_won[0], env.bids[0]) if is_last else 0.0
        # DMC return: shaped + terminal (no discount for Plump short horizon)
        play_returns.append(shaped + term)

    return learner_bids, learner_plays, bid_returns, play_returns, env


def train_restart():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DMC] device {device}, vocab {VOCAB}, feat {FEAT_DIM}")
    cfg = PlumpTransformerConfig(d_model=64, n_blocks=2, n_heads=4, ffn_hidden=256, hand_hidden=128, q_hidden=256)
    # 2-block 64d for 5h restart — swap to 4x128 for long-term via config
    model = PlumpTransformer(cfg).to(device)
    # Bid and play share encoder but separate heads — single model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ── RESUME: pick up exactly where the last session stopped ───────────────
    # Session counter survives restarts via plump_transformer_session.txt
    start_ep = 0
    if os.path.exists("plump_transformer_latest.pt"):
        try:
            model.load_state_dict(torch.load("plump_transformer_latest.pt", map_location=device))
            print("[resume] loaded plump_transformer_latest.pt — continuing, not restarting")
        except Exception as e:
            print(f"[!] resume failed ({e}) — starting fresh")
    if os.path.exists("plump_transformer_session.txt"):
        try:
            start_ep = int(Path("plump_transformer_session.txt").read_text().strip())
            print(f"[resume] episode counter at {start_ep:,}")
        except Exception:
            start_ep = 0

    # Replay buffers also persist so DMC doesn't lose its experience on rerun
    bid_buffer: list[tuple] = []
    play_buffer: list[tuple] = []
    buf_path_b = Path("plump_transformer_buf_bid.npz")
    buf_path_p = Path("plump_transformer_buf_play.npz")

    def _save_buffers():
        if bid_buffer:
            np.savez(buf_path_b,
                toks=np.stack([b[0] for b in bid_buffer]).astype(np.int16),
                feats=np.stack([b[1] for b in bid_buffer]),
                act=np.array([b[2] for b in bid_buffer], dtype=np.int16),
                ret=np.array([b[3] for b in bid_buffer], dtype=np.float32))
        if play_buffer:
            np.savez(buf_path_p,
                toks=np.stack([p[0] for p in play_buffer]).astype(np.int16),
                feats=np.stack([p[1] for p in play_buffer]),
                act=np.array([p[2] for p in play_buffer], dtype=np.int16),
                ret=np.array([p[3] for p in play_buffer], dtype=np.float32))

    def _load_buffers():
        for path, buf, mk in ((buf_path_b, bid_buffer, "bid"), (buf_path_p, play_buffer, "play")):
            if path.exists():
                z = np.load(path)
                for i in range(len(z["act"])):
                    buf.append((z["toks"][i].astype(np.int64), z["feats"][i], int(z["act"][i]), float(z["ret"][i])))
                print(f"[resume] {mk} buffer restored: {len(buf)} samples")

    _load_buffers()
    # Save buffers at most every 10k episodes (writes ~100MB; more often would stall the GPU)
    buf_last_save = [start_ep]

    # League includes old DQN champions if present
    dqn_paths = []
    if os.path.exists("plump_bid_model_champion.pt"):
        dqn_paths.append(("plump_bid_model_champion.pt", "plump_play_model_champion.pt"))
    if os.path.exists("plump_bid_model_best_v2.pt"):
        dqn_paths.append(("plump_bid_model_best_v2.pt", "plump_play_model_best_v2.pt"))
    league = League(dqn_paths if dqn_paths else None)
    print(f"[league] {len(league.members)} members: {[n for n,_ in league.members]}")

    env = PlumpEnv()

    episodes = start_ep + 800_000  # runway covers 6.5h+ sessions (~25-30 games/s); Ctrl+C anytime
    batch = 128
    best = -1e9
    if os.path.exists("plump_transformer_best_score.txt"):
        try:
            best = float(Path("plump_transformer_best_score.txt").read_text().strip())
        except Exception:
            best = -1e9

    try:
        for ep in range(start_ep + 1, episodes + 1):
            # mild exploration decay, floored so league stays diverse forever
            epsilon = max(0.05, 0.20 * (0.99995 ** ep))
            b_dec, p_dec, b_ret, p_ret, env_out = play_one_game_transformer(env, model, league, device, epsilon)

            # push to buffers (tuples: toks, feats, action, return)
            for d, ret in zip(b_dec, b_ret):
                bid_buffer.append((d["toks"], d["feats"], d["action"], ret))
                if len(bid_buffer) > 50000:
                    bid_buffer.pop(0)
            for d, ret in zip(p_dec, p_ret):
                play_buffer.append((d["toks"], d["feats"], d["action"], ret))
                if len(play_buffer) > 100000:
                    play_buffer.pop(0)

            # ── LEARN every 16 games ────────────────────────────────────────
            if ep % 16 == 0 and len(play_buffer) >= batch:
                batch_items = random.sample(play_buffer, batch)
                toks_batch = torch.from_numpy(np.array([x[0] for x in batch_items], dtype=np.int64)).to(device)
                feats_batch = torch.tensor(np.stack([x[1] for x in batch_items]), dtype=torch.float32, device=device)
                actions = torch.tensor([x[2] for x in batch_items], dtype=torch.long, device=device)
                returns = torch.tensor([x[3] for x in batch_items], dtype=torch.float32, device=device)
                model.train()
                q = model(toks_batch, feats_batch, phase="play")
                q_taken = q.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = nn.MSELoss()(q_taken, returns)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if ep % 16 == 0 and len(bid_buffer) >= batch:
                batch_items = random.sample(bid_buffer, batch)
                toks_batch = torch.from_numpy(np.array([x[0] for x in batch_items], dtype=np.int64)).to(device)
                feats_batch = torch.tensor(np.stack([x[1] for x in batch_items]), dtype=torch.float32, device=device)
                actions = torch.tensor([x[2] for x in batch_items], dtype=torch.long, device=device)
                returns = torch.tensor([x[3] for x in batch_items], dtype=torch.float32, device=device)
                model.train()
                q = model(toks_batch, feats_batch, phase="bid")
                q_taken = q.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = nn.MSELoss()(q_taken, returns)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # ── EVAL + CHECKPOINT every 1000 games ──────────────────────────
            if ep % 1000 == 0:
                wins = 0
                total = 0
                for _ in range(200):
                    _, _, _, _, env_e = play_one_game_transformer(PlumpEnv(), model, league, device, epsilon=0.0)
                    total += terminal_reward(env_e.tricks_won[0], env_e.bids[0])
                    if env_e.tricks_won[0] == env_e.bids[0]:
                        wins += 1
                winrate = wins / 200 * 100
                avg = total / 200
                print(f"Ep {ep:6d} | win {winrate:5.1f}% avg {avg:+5.2f} | buf {len(play_buffer)}/{len(bid_buffer)}")
                if avg > best:
                    best = avg
                    Path("plump_transformer_best_score.txt").write_text(f"{best:.4f}")
                    torch.save(model.state_dict(), "plump_transformer_best.pt")
                    print(f"  -> new best {best:.2f} saved")
                torch.save(model.state_dict(), "plump_transformer_latest.pt")
                Path("plump_transformer_session.txt").write_text(str(ep))
                if ep - buf_last_save[0] >= 10_000:
                    _save_buffers()
                    buf_last_save[0] = ep

    except KeyboardInterrupt:
        print(f"\n[!] interrupted at ep {ep:,} — saving full state (weights + buffers + counter)...")
        torch.save(model.state_dict(), "plump_transformer_latest.pt")
        Path("plump_transformer_session.txt").write_text(str(ep))
        _save_buffers()
        print("[+] saved. Next run continues exactly here.")

    print("done restart DMC+Transformer+league — old DQN was in league as sparring partner")


if __name__ == "__main__":
    train_restart()
