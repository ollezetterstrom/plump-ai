# train_transformer.py — restart from scratch, correct algorithm.
# Transformer + DMC + league (old DQN in pool as sparring partner).
# Vibe-proof: config in plump/config/settings.py, logic in plump/training/train_dmc.py

if __name__ == "__main__":
    from plump.training.train_dmc import train_restart

    train_restart()
