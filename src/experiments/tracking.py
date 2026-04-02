import csv
from pathlib import Path
from datetime import datetime
from src.utils.io import ensure_dirs, copy_file, save_json

def new_run_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"outputs/runs/{ts}"
    ensure_dirs(run_dir)
    return run_dir

def snapshot_config(config_path: str, run_dir: str) -> None:
    copy_file(config_path, f"{run_dir}/config.yaml")

def write_run_summary(run_dir: str, summary: dict) -> None:
    save_json(f"{run_dir}/summary.json", summary)

def append_runs_csv(summary: dict) -> None:
    ensure_dirs("outputs/runs")
    csv_path = Path("outputs/runs/runs.csv")
    fieldnames = sorted(summary.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(summary)
