"""Shared run-logging helper. No project dependencies, so every script can
use it without pulling in sklearn (common.py) or requiring a database
connection (db.py)."""

import sys
from datetime import datetime
from pathlib import Path


class Tee:
    """Mirrors writes to every stream it wraps (e.g. the real console plus
    a log file), so redirecting sys.stdout/sys.stderr through one of these
    logs a full run without touching any of the print() calls above."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_with_file_logging(log_dir, prefix, main_fn):
    """Run main_fn() with stdout/stderr mirrored to a timestamped log file
    under log_dir, restoring the real streams afterward."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{prefix}_run_{datetime.now():%Y%m%d_%H%M%S}.log"

    real_stdout, real_stderr = sys.stdout, sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_f:
        sys.stdout = Tee(real_stdout, log_f)
        sys.stderr = Tee(real_stderr, log_f)

        try:
            print(f"Logging full run output to: {log_path}")
            main_fn()
        finally:
            sys.stdout = real_stdout
            sys.stderr = real_stderr
