from pathlib import Path
import subprocess
import sys
import time
import csv


SCRIPTS = [
    "summary_stats.py",
    "xgboost_engineering.py",
    "xgboost_train.py",
    "lstm_engineering.py",
    "lstm_train.py",
    "probability_plot.py",
]

LOG_FILE = Path("run_times.csv")


def run_script(script):
    print("\n" + "=" * 80)
    print(f"Running: {script}")
    print("=" * 80)

    start = time.time()

    result = subprocess.run([sys.executable, script])

    end = time.time()
    seconds = end - start

    status = "success" if result.returncode == 0 else "failed"

    print(f"Finished: {script}")
    print(f"Status: {status}")
    print(f"Time: {seconds / 60:.2f} minutes")

    return {
        "script": script,
        "status": status,
        "return_code": result.returncode,
        "seconds": round(seconds, 2),
        "minutes": round(seconds / 60, 2),
    }


def main():
    results = []

    total_start = time.time()

    for script in SCRIPTS:
        if not Path(script).exists():
            print(f"\nMissing script, skipping: {script}")

            results.append({
                "script": script,
                "status": "missing",
                "return_code": "",
                "seconds": "",
                "minutes": "",
            })

            continue

        result = run_script(script)
        results.append(result)

        if result["status"] == "failed":
            print(f"\nStopping because {script} failed.")
            break

    total_seconds = time.time() - total_start

    results.append({
        "script": "TOTAL",
        "status": "complete",
        "return_code": "",
        "seconds": round(total_seconds, 2),
        "minutes": round(total_seconds / 60, 2),
    })

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["script", "status", "return_code", "seconds", "minutes"],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print("Run complete")
    print("=" * 80)
    print(f"Saved timing log to: {LOG_FILE}")
    print(f"Total time: {total_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    main()