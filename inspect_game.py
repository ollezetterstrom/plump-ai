import torch
import plump_engine
import play
from plump.config import BEST_CHECKPOINT, LATEST_CHECKPOINT, ROUND_SCHEDULE
from plump.checkpoint import load_model

SUIT_NAMES = ("H", "S", "D", "C")
RANK_NAMES = {2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10", 11:"J", 12:"Q", 13:"K", 14:"A"}

def fmt_card(c: int) -> str:
    return f"{SUIT_NAMES[c // 13]}{RANK_NAMES[c % 13 + 2]}"

def fmt_hand(cards) -> str:
    sorted_cards = sorted(cards, key=lambda c: (c // 13, -(c % 13)))
    return " ".join(fmt_card(c) for c in sorted_cards)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = BEST_CHECKPOINT if BEST_CHECKPOINT.exists() else LATEST_CHECKPOINT
    model, meta = load_model(ckpt, device)
    print(f"=== AI SELF-PLAY LOG ({ckpt.name}, rounds_seen={meta.get('rounds_seen')}, best_score={meta.get('best_score'):.3f}) ===")

    settings = play.PlaySettings(autoplay=True, use_search=True, worlds=32, model_available=True)
    env = plump_engine.PyFastEnv(seed=2026)
    totals = [0, 0, 0, 0]
    plumps = [0, 0, 0, 0]

    for r_idx, round_cards in enumerate(ROUND_SCHEDULE):
        start_bidder = r_idx % 4
        dealer = (start_bidder + 3) % 4
        env.new_round(round_cards, start_bidder)
        st = env.state()

        print(f"\n--- ROUND {r_idx+1:02d}/20 ({round_cards} cards | Dealer: P{dealer} | First Bidder: P{start_bidder}) ---")
        for p in range(4):
            print(f"  P{p} Hand: [{fmt_hand(st['hands'][p])}]")

        bid_logs = []
        while env.phase == "Bidding":
            p = int(env.current_player)
            forb = env.forbidden_bid()
            bid, dec = play.choose_bid(env, model, settings, device)
            env.step_bid(bid)
            forb_str = f" (forb={forb})" if forb is not None else ""
            bid_logs.append(f"P{p}:{bid}{forb_str} [{dec.extra}]")
        leader = int(env.get_leader())
        print(f"  Bids: " + " -> ".join(bid_logs) + f" | Leader: P{leader} (bid {env.bids[leader]})")

        while env.phase == "Playing":
            t_num = int(env.state()["current_trick"]) + 1
            trick_plays = []
            winner = None
            while winner is None and env.phase == "Playing":
                p = int(env.current_player)
                card, dec = play.choose_card(env, model, settings, device)
                winner = env.step_play(card)
                trick_plays.append(f"P{p}:{fmt_card(card)}")
            print(f"  T{t_num:02d}: " + " -> ".join(trick_plays) + f"  => P{winner} wins")

        pts = [int(env.display_points_for(p)) for p in range(4)]
        res_logs = []
        for p in range(4):
            b = int(env.bids[p])
            w = int(env.tricks_won[p])
            if w != b:
                plumps[p] += 1
            totals[p] += pts[p]
            status = "OK" if w == b else "PLUMP"
            res_logs.append(f"P{p}: {w}/{b} ({status}, +{pts[p]} -> {totals[p]})")
        print("  Result: " + " | ".join(res_logs))

    print("\n=== FINAL SCOREBOARD ===")
    for p in range(4):
        print(f"  P{p}: {totals[p]} pts ({plumps[p]} plumps)")

if __name__ == "__main__":
    main()
