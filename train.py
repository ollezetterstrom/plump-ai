# train.py — shim for backward compat. Real logic lives in plump/*
# Vibe-proof: hyperparams in plump/config/settings.py, runner in plump/training/runner.py
import os
import random
import torch
import torch.optim as optim

from plump.env import PlumpEnv
from plump.encode import encode_state_278, encode_state_294, mask_actions
from plump.env.cards import INDEX_TO_CARD
from plump.models import DuelingQNetwork
from plump.training.buffers import ReplayBuffer
from plump.training.runner import run_one_game, pick_action
from plump.config.settings import TRAIN as CFG, HOUSE, terminal_reward, display_points
from plump.training.rewards import shaped_reward

# Re-export for old imports (play.py does `from train import DuelingQNetwork`)
__all__ = [
    "PlumpEnv",
    "DuelingQNetwork",
    "ReplayBuffer",
    "run_one_game",
    "terminal_reward",
    "display_points",
    "shaped_reward",
]

# Hyperparams from single source
EPISODES = CFG.episodes
BATCH_SIZE = CFG.batch_size
GAMMA = CFG.gamma
TAU = CFG.tau
LEARN_EVERY = CFG.learn_every
CPU_SYNC_EVERY = CFG.cpu_sync_every
EVAL_EVERY = CFG.eval_every
EVAL_GAMES = CFG.eval_games
LR = CFG.lr
EPSILON_START = CFG.epsilon_start
EPSILON_MIN = CFG.epsilon_min
EPSILON_DECAY = CFG.epsilon_decay
EPSILON_RESTART_EVERY = CFG.epsilon_restart_every
EPSILON_RESTART_VALUE = CFG.epsilon_restart_value
TRAIN_RANDOM_CHANCE = CFG.train_random_chance
BID_WARMUP = CFG.bid_warmup
BID_BUFFER_SIZE = CFG.bid_buffer
PLAY_BUFFER_SIZE = CFG.play_buffer


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def update_network(model, target_model, optimizer, buffer, device):
    states, actions, rewards, next_states, dones, masks = buffer.sample(BATCH_SIZE)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    masks = masks.to(device)
    q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_online = model(next_states)
        next_q_online[~masks] = -1e9
        best_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = target_model(next_states)
        next_max = next_q_target.gather(1, best_actions).squeeze(1)
        targets = rewards + GAMMA * next_max * (1.0 - dones)
    loss = torch.nn.SmoothL1Loss()(q_vals, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        print("[CPU] No GPU detected — training will be slow.")

    champ_bid = DuelingQNetwork(278, 11).to(cpu_device)
    champ_play = DuelingQNetwork(278, 52).to(cpu_device)
    champ_bid.eval()
    champ_play.eval()
    champ_loaded = False

    bid_model = DuelingQNetwork(278, 11).to(device)
    bid_target = DuelingQNetwork(278, 11).to(device)
    bid_target.load_state_dict(bid_model.state_dict())
    play_model = DuelingQNetwork(294, 52).to(device)
    play_target = DuelingQNetwork(294, 52).to(device)
    play_target.load_state_dict(play_model.state_dict())

    if os.path.exists("plump_bid_model_champion.pt") and os.path.exists("plump_play_model_champion.pt"):
        champ_bid.load_state_dict(torch.load("plump_bid_model_champion.pt", map_location=cpu_device))
        champ_play.load_state_dict(torch.load("plump_play_model_champion.pt", map_location=cpu_device))
        champ_loaded = True
        print("[+] Champion opponents loaded (278-dim, fixed).")
    else:
        print("[!] No champion models found — eval opponents play randomly.")

    if os.path.exists("plump_bid_model_latest_v2.pt") and os.path.exists("plump_play_model_latest_v2.pt"):
        bid_model.load_state_dict(torch.load("plump_bid_model_latest_v2.pt", map_location=device))
        bid_target.load_state_dict(bid_model.state_dict())
        play_model.load_state_dict(torch.load("plump_play_model_latest_v2.pt", map_location=device))
        play_target.load_state_dict(play_model.state_dict())
        print("[+] Resumed from latest v2 checkpoints.")
    elif champ_loaded:
        bid_model.load_state_dict(torch.load("plump_bid_model_champion.pt", map_location=device))
        bid_target.load_state_dict(bid_model.state_dict())
        print("[+] Bid model warm-started from champion. Play model starts fresh.")
    else:
        print("[!] Training both networks completely from scratch.")

    bid_model_cpu = DuelingQNetwork(278, 11).to(cpu_device)
    play_model_cpu = DuelingQNetwork(294, 52).to(cpu_device)
    bid_model_cpu.load_state_dict(bid_model.state_dict())
    play_model_cpu.load_state_dict(play_model.state_dict())
    bid_model_cpu.eval()
    play_model_cpu.eval()

    bid_optimizer = optim.Adam(bid_model.parameters(), lr=LR)
    play_optimizer = optim.Adam(play_model.parameters(), lr=LR)
    bid_buffer = ReplayBuffer(BID_BUFFER_SIZE)
    play_buffer = ReplayBuffer(PLAY_BUFFER_SIZE)

    epsilon = EPSILON_START
    best_winrate = 0.0
    learn_steps = 0
    env = PlumpEnv()

    print("\n" + "=" * 72)
    print(f"  PLUMP AI - v2 HOUSE RULES (0=5, leader=highest)")
    print(f"  Bid warmup    : {BID_WARMUP:,} episodes (then always trains)")
    print(f"  Epsilon       : {EPSILON_START} -> {EPSILON_MIN}, restart to {EPSILON_RESTART_VALUE} every {EPSILON_RESTART_EVERY:,} eps")
    print(f"  Buffers       : bid={BID_BUFFER_SIZE:,}  play={PLAY_BUFFER_SIZE:,}")
    print(f"  Champion eval : {'yes (278-dim)' if champ_loaded else 'no (random)'}")
    print("=" * 72 + "\n")

    try:
        for ep in range(1, EPISODES + 1):
            b_trans, p_trans, _ = run_one_game(env, bid_model_cpu, play_model_cpu, champ_bid if champ_loaded else None, champ_play if champ_loaded else None, epsilon, cpu_device, is_eval=False)
            for t in b_trans:
                bid_buffer.push(*t)
            for t in p_trans:
                play_buffer.push(*t)
            if ep % LEARN_EVERY == 0:
                if len(play_buffer) >= BATCH_SIZE:
                    update_network(play_model, play_target, play_optimizer, play_buffer, device)
                    soft_update(play_target, play_model, TAU)
                if ep > BID_WARMUP and len(bid_buffer) >= BATCH_SIZE:
                    update_network(bid_model, bid_target, bid_optimizer, bid_buffer, device)
                    soft_update(bid_target, bid_model, TAU)
                learn_steps += 1
                if learn_steps % CPU_SYNC_EVERY == 0:
                    bid_model_cpu.load_state_dict(bid_model.state_dict())
                    play_model_cpu.load_state_dict(play_model.state_dict())
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            if ep % EPSILON_RESTART_EVERY == 0:
                epsilon = EPSILON_RESTART_VALUE
                print(f"\n[!] Epsilon restarted -> {epsilon:.2f}  (ep {ep:,})")
            if ep % EVAL_EVERY == 0:
                bid_model_cpu.load_state_dict(bid_model.state_dict())
                play_model_cpu.load_state_dict(play_model.state_dict())
                wins = 0
                total_score = 0.0
                for _ in range(EVAL_GAMES):
                    _, _, score = run_one_game(env, bid_model_cpu, play_model_cpu, champ_bid if champ_loaded else None, champ_play if champ_loaded else None, 0.0, cpu_device, is_eval=True)
                    if score > 0:
                        wins += 1
                    total_score += score
                winrate = (wins / EVAL_GAMES) * 100
                avg_score = total_score / EVAL_GAMES
                bar_fill = min(20, int((winrate / 50.0) * 20))
                bar = "#" * bar_fill + "." * (20 - bar_fill)
                bid_status = "TRAINING" if ep > BID_WARMUP else f"WARMUP ({BID_WARMUP - ep:,} left)"
                torch.save(bid_model.state_dict(), "plump_bid_model_latest_v2.pt")
                torch.save(play_model.state_dict(), "plump_play_model_latest_v2.pt")
                marker = ""
                if winrate > best_winrate:
                    best_winrate = winrate
                    marker = " <- NEW BEST"
                    torch.save(bid_model.state_dict(), "plump_bid_model_best_v2.pt")
                    torch.save(play_model.state_dict(), "plump_play_model_best_v2.pt")
                print(f"Ep {ep:>8,} | Win {winrate:>5.1f}% [{bar}] | AvgScore {avg_score:>+6.2f} | eps {epsilon:.3f} | Bid: {bid_status}{marker}")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted. Saving current state...")
        torch.save(bid_model.state_dict(), "plump_bid_model_interrupted_v2.pt")
        torch.save(play_model.state_dict(), "plump_play_model_interrupted_v2.pt")
        print("[+] Saved. Resume by renaming interrupted files to latest_v2.")


if __name__ == "__main__":
    train()
