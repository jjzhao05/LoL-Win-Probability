import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import csv
from datetime import datetime


# Each entry is one pipeline step. `inputs` are files that, if newer than
# the step's outputs, mean the step is stale and must rerun (the step's own
# source file is always included). `outputs` are what the step produces; a
# step is skipped only when every output exists (non-empty, for a
# directory) and no input is newer than the oldest output.
PIPELINE = [
    {
        "script": "summary_stats.py",
        "inputs": ["summary_stats.py", "db.py"],
        "outputs": [
            "data/summary_stats.csv",
            "data/champion_stats.csv",
            "data/ban_stats.csv",
            "data/role_stats.csv",
            "data/objective_impact.csv",
            "data/feature_correlations.csv",
            "data/patch_stats.csv",
            "data/notable_games.csv",
            "data/summoner_spell_stats.csv",
            "data/rune_style_stats.csv",
            "data/keystone_stats.csv",
        ],
    },
    {
        "script": "xgboost_engineering.py",
        "inputs": ["xgboost_engineering.py", "common.py", "db.py"],
        "outputs": ["xgb_engineered/xgb_clean_dataset.parquet"],
    },
    {
        "script": "xgboost_train.py",
        "inputs": ["xgboost_train.py", "common.py", "xgb_engineered/xgb_clean_dataset.parquet"],
        "outputs": [
            "models/xgb_model.json",
            "results/xgb_predictions.parquet",
            "results/xgb_grid_search_results.csv",
            "results/xgb_feature_importances.csv",
            "figures/xgb_calibration_curve.png",
            "figures/xgb_feature_importance.png",
            "results/shared_split_ids.npz",
        ],
    },
    {
        "script": "logistic_regression_train.py",
        "inputs": [
            "logistic_regression_train.py",
            "common.py",
            "xgb_engineered/xgb_clean_dataset.parquet",
            "results/shared_split_ids.npz",
        ],
        "outputs": [
            "models/logreg_model.joblib",
            "results/logreg_predictions.parquet",
            "results/logreg_grid_search_results.csv",
            "results/logreg_feature_importances.csv",
            "figures/logreg_calibration_curve.png",
            "figures/logreg_feature_importance.png",
        ],
    },
    {
        "script": "probability_plot.py",
        "inputs": [
            "probability_plot.py",
            "common.py",
            "results/xgb_predictions.parquet",
            "results/logreg_predictions.parquet",
        ],
        "outputs": ["figures/xgb_vs_logreg_labeled_plots"],
    },
]

LOG_FILE = Path("results/run_times.csv")

LOG_DIR = Path("logs")
RUN_LOG_FILE = LOG_DIR / f"main_run_{datetime.now():%Y%m%d_%H%M%S}.log"


def oldest_output_mtime(output):
    """None if the output is missing (or an empty directory); otherwise the
    mtime of the output itself, or the mtime of its oldest file if it's a
    directory of files (e.g. probability_plot.py's plot folder)."""
    path = Path(output)

    if path.is_dir():
        files = [f for f in path.iterdir() if f.is_file()]
        if not files:
            return None
        return min(f.stat().st_mtime for f in files)

    if path.exists():
        return path.stat().st_mtime

    return None


def is_up_to_date(step):
    output_mtimes = [oldest_output_mtime(o) for o in step["outputs"]]

    if not output_mtimes or any(m is None for m in output_mtimes):
        return False

    oldest_output = min(output_mtimes)

    for inp in step["inputs"]:
        inp_path = Path(inp)
        if inp_path.exists() and inp_path.stat().st_mtime > oldest_output:
            return False

    return True


def run_script(script, log_f):
    header = "\n" + "=" * 80 + f"\nRunning: {script}\n" + "=" * 80
    print(header)
    log_f.write(header + "\n")
    log_f.flush()

    start = time.time()

    # `-u` (and PYTHONUNBUFFERED) force unbuffered stdout: piping a child's
    # stdout switches Python to full block buffering, so without this a
    # long-running step's print() calls can sit unflushed for minutes.
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )

    for line in process.stdout:
        print(line, end="")
        log_f.write(line)
        log_f.flush()

    process.wait()

    end = time.time()
    seconds = end - start

    status = "success" if process.returncode == 0 else "failed"

    footer = (
        f"Finished: {script}\n"
        f"Status: {status}\n"
        f"Time: {seconds / 60:.2f} minutes"
    )
    print(footer)
    log_f.write(footer + "\n")
    log_f.flush()

    return {
        "script": script,
        "status": status,
        "return_code": process.returncode,
        "seconds": round(seconds, 2),
        "minutes": round(seconds / 60, 2),
    }


def skip_result(script):
    msg = f"\nSkipping (up to date): {script}"
    print(msg)

    return {
        "script": script,
        "status": "skipped",
        "return_code": "",
        "seconds": 0,
        "minutes": 0,
    }, msg


def parse_args():
    parser = argparse.ArgumentParser(description="Run the win-probability modeling pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun every step, ignoring cached outputs (old behavior).",
    )
    parser.add_argument(
        "--rerun",
        nargs="+",
        metavar="SCRIPT",
        default=[],
        help=(
            "Rerun only these steps even if their outputs look up to date "
            "(e.g. --rerun logistic_regression_train.py). Everything else "
            "is still skipped when cached."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    force_rerun = set(args.rerun)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    total_start = time.time()

    with open(RUN_LOG_FILE, "w", encoding="utf-8") as log_f:
        start_msg = f"Run started: {datetime.now().isoformat()}"
        print(f"Logging full run output to: {RUN_LOG_FILE}")
        log_f.write(start_msg + "\n")
        log_f.flush()

        for step in PIPELINE:
            script = step["script"]

            if not Path(script).exists():
                msg = f"\nMissing script, skipping: {script}"
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()

                results.append({
                    "script": script,
                    "status": "missing",
                    "return_code": "",
                    "seconds": "",
                    "minutes": "",
                })

                continue

            should_force = args.force or script in force_rerun

            if not should_force and is_up_to_date(step):
                result, msg = skip_result(script)
                log_f.write(msg + "\n")
                log_f.flush()
                results.append(result)
                continue

            result = run_script(script, log_f)
            results.append(result)

            if result["status"] == "failed":
                msg = f"\nStopping because {script} failed."
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
                break

        total_seconds = time.time() - total_start

        results.append({
            "script": "TOTAL",
            "status": "complete",
            "return_code": "",
            "seconds": round(total_seconds, 2),
            "minutes": round(total_seconds / 60, 2),
        })

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["script", "status", "return_code", "seconds", "minutes"],
            )

            writer.writeheader()
            writer.writerows(results)

        summary = (
            "\n" + "=" * 80 + "\n"
            "Run complete\n" + "=" * 80 + "\n"
            f"Saved timing log to: {LOG_FILE}\n"
            f"Total time: {total_seconds / 60:.2f} minutes\n"
            f"Full run log: {RUN_LOG_FILE}"
        )
        print(summary)
        log_f.write(summary + "\n")


if __name__ == "__main__":
    main()
