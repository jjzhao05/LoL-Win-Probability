# League of Legends Win Probability Model

Does win probability in League of Legends really just reduce to who has more
gold, or does objective control (dragons, heralds, barons, towers) carry
predictive value the economy can't fully substitute for?

This project builds two win-probability models — a linear logistic
regression baseline and a nonlinear XGBoost tree ensemble — trained on the
same rank-stratified, single-patch dataset and the same held-out match
split, then answers that question directly with a feature-ablation study
(`ablation.py`): retrain each model with every gold/XP/level feature removed,
then again with every objective/tower feature removed, and see how much
predictive power survives each time. The two model families exist to check
whether the answer is a property of the data or an artifact of how one
architecture happens to split on collinear features — if a plain linear
model and a tree ensemble agree, the finding is a lot more trustworthy than
either one alone.

The ablation goes a step further than a single point estimate: it bootstraps
a 95% confidence interval on each AUC gap (resampled at the match level, so
correlated per-minute rows within a match are resampled together), and
breaks the objectives ablation down by how close the game state is
(`|team_total_gold_diff|`) to check whether objective control's independent
value concentrates in close games, where there's no economic lead yet to
fall back on.

## Running the pipeline

```
python main.py
```

Runs `combine.py -> summary_stats.py -> xgboost_engineering.py ->
xgboost_train.py -> logistic_regression_train.py -> probability_plot.py` in
order and logs timing to `results/run_times.csv`. `collect_data.py` is
intentionally excluded (it's a slow, rate-limited Riot API pull meant to be
run manually to populate `data/`). `ablation.py` is also run separately once
both models exist, since it retrains both of them again per ablation variant.

All scripts assume they're run from the repo root, since every path they read
or write is relative to it.

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
├── logistic_regression_train.py  # trains + grid-searches the logistic regression model
├── ablation.py                # gold/XP/level and objective feature ablations, for both models
├── probability_plot.py        # per-match XGBoost-vs-logistic-regression probability charts
├── app.py                     # interactive dashboard (streamlit run app.py)
├── main.py                    # runs the pipeline end to end (see SCRIPTS list in main.py)
├── common.py                  # shared constants + eval helpers used across scripts
│
├── data/                      # raw + combined match data (per-rank snapshots, full_dataset.parquet)
├── lol_data/                  # small hand-authored ID -> name lookup tables (champions, runes, spells)
├── xgb_engineered/            # engineered feature table used by both models
│
├── models/                    # trained model artifacts (xgb_model.json, logreg_model.joblib)
├── results/                   # metrics, predictions, and the shared train/val/test split
├── figures/                   # calibration curves, feature importance plots, per-match charts
└── reports/                   # capstone report PDFs
```

> `app.py` (`streamlit run app.py`) has two pages: "Explore a Game" (a real
> held-out match's XGBoost vs. logistic regression win-probability curves,
> plus a custom-scenario predictor comparing both models on macro slider
> inputs) and "Findings" (the ablation study's AUC-by-variant chart with
> bootstrap CIs, the closeness-stratified breakdown, calibration curves,
> feature importances, and per-minute-bucket performance).

## Data collection

`collect_data.py` stratifies ranked solo/duo matches by skill bracket (10
tiers, IRON through CHALLENGER) and is pinned to a single game patch
(`TARGET_PATCH`, set at the top of the script). Patches change champion
balance and item economics, so pooling matches across several of them would
confound "which rank" with "which patch was live" — pinning to one patch
keeps rank the only thing varying across the dataset. Set `TARGET_PATCH` to
whatever patch is live before starting a collection run.

## Setup

```
pip install -r requirements.txt
```

Requires a `.env` file with a Riot API key for `collect_data.py`.
