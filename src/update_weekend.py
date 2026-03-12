import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

from predict import predict_session
from config import OUTPUTS_DIR, DEFAULT_PREDICT_YEAR, DEFAULT_PREDICT_ROUND
from detect_round import detect_current_round
from detect_stage import detect_stage
from simulate_race import run_monte_carlo

HISTORY_DIR = OUTPUTS_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def clean_stage_name(stage: str) -> str:
    return (
        stage.strip()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
    )


def snapshot_file(path: Path, timestamp: str, stage: str):
    if not path.exists():
        return None

    safe_stage = clean_stage_name(stage)
    snapshot_name = f"{path.stem}__{safe_stage}__{timestamp}{path.suffix}"
    snapshot_path = HISTORY_DIR / snapshot_name
    shutil.copy2(path, snapshot_path)
    return snapshot_path


def write_metadata(
    year: int,
    round_number: int,
    sessions_available: list[str],
    timestamp: str,
    stage: str
):
    metadata = {
        "year": year,
        "round": round_number,
        "last_updated": timestamp,
        "stage": stage,
        "sessions_available": sessions_available,
    }
    out_file = OUTPUTS_DIR / "metadata.json"
    out_file.write_text(json.dumps(metadata, indent=2))


def update_weekend(year: int, round_number: int, stage: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sessions_available = []

    for session_code in ["Q", "R", "SQ", "S"]:
        try:
            print(f"Running prediction for {session_code}...")
            predict_session(year, round_number, session_code)

            out_file = OUTPUTS_DIR / f"{year}_{round_number}_{session_code}_predictions.csv"
            snapshot_path = snapshot_file(out_file, timestamp, stage)

            if snapshot_path is not None:
                print(f"Saved snapshot: {snapshot_path.name}")

            sessions_available.append(session_code)

        except Exception as e:
            print(f"Skipped {session_code}: {e}")

    # Run race simulation after predictions
    try:
        print("Running race simulation...")
        run_monte_carlo(year, round_number, n_sims=5000)
    except Exception as e:
        print(f"Skipped race simulation: {e}")
    
    write_metadata(year, round_number, sessions_available, timestamp, stage)
    print(f"Weekend outputs updated for stage: {stage}")


if __name__ == "__main__":
    year = DEFAULT_PREDICT_YEAR
    round_number = detect_current_round(year)
    stage = None

    if len(sys.argv) >= 3:
        year = int(sys.argv[1])
        round_number = int(sys.argv[2])

    if len(sys.argv) >= 4:
        stage = sys.argv[3]

    if stage is None:
        stage = detect_stage(year, round_number)

    update_weekend(year, round_number, stage)