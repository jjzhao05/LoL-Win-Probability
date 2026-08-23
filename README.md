# League of Legends Win Probability Model

Predicts blue-side (team 100) win probability at every in-game minute from
timeline snapshots, using both an XGBoost model and an every-minute LSTM,
trained and compared on the same held-out match split.

## Project layout

All pipeline scripts live flat in the repo root (no nested script folders).
Data, models, and generated outputs are split into their own top-level
folders so the root stays readable:

```
.
├── collect_data.py            # pulls raw match timelines from the Riot API (manual/one-off)
├── combine.py                 # merges data/<RANK>_snapshots_parquet/ -> data/full_dataset.parquet
├── summary_stats.py           # descriptive stats -> data/*.csv
├── xgboost_engineering.py     # data/full_dataset.parquet -> xgb_engineered/xgb_clean_dataset.parquet
├── xgboost_train.py           # trains + grid-searches the XGBoost model
├── lstm_engineering.py        # data/full_dataset.parquet -> results/lstm_data.npz
├── lstm_train.py              # trains + grid-searches the LSTM model
├── ablation.py                # gold/XP/level and objective feature ablations
├── probability_plot.py        # per-match XGBoost-vs-LSTM probability charts
├── app.py                     # interactive dashboard (streamlit run app.py)
├── main.py                    # runs the pipeline end to end (see SCRIPTS list in main.py)
├── common.py                  # shared constants + eval helpers used across scripts
├── compare.ipynb              # notebook scratch space
│
├── data/                      # raw + combined match data (per-rank snapshots, full_dataset.parquet)
├── lol_data/                  # small hand-authored ID -> name lookup tables (champions, runes, spells)
├── xgb_engineered/            # engineered feature table used only by the XGBoost side
│
├── models/                    # trained model artifacts (lstm_model.pt, xgb_model.json)
├── results/                   # metrics, predictions, and the shared train/val/test split
├── figures/                   # calibration curves, feature importance plots, per-match charts
└── reports/                   # capstone report PDFs
```

## Running the pipeline

```
python main.py
```

Runs `combine.py -> summary_stats.py -> xgboost_engineering.py -> xgboost_train.py
-> lstm_engineering.py -> lstm_train.py -> probability_plot.py` in order and logs
timing to `results/run_times.csv`. `collect_data.py` is intentionally excluded
(it's a slow, rate-limited Riot API pull meant to be run manually to populate
`data/`). `ablation.py` is also run separately once both models exist.

All scripts assume they're run from the repo root, since every path they read
or write is relative to it.

## Setup

```
pip install -r requirements.txt
```

Requires a `.env` file with a Riot API key for `collect_data.py`.
