# Training

House 5h on RX 9060 XT: ~35 eps/s → ~630k games.

```powershell
python -u train.py   # resumes plump_bid_model_latest_v2.pt, saves best_v2.pt every 1000
# Ctrl+C saves interrupted_v2.pt
python play.py       # vs AI, 5pt + leader
```

Config: `plump/config/settings.yaml` (episodes, batch, lr, epsilon). Long-term: see `plump/models/transformer.py` stub and research in `C:\Users\ozett\AppData\Local\Temp\opencode`.
