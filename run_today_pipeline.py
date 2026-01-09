from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PY = sys.executable  # <-- THIS guarantees venv python is used


STEPS = [
    (
        "Update current player → team map",
        [PY, "data/raw/fetch_current_player_teams.py"],
    ),
    (
        "Fetch today's points props (FanDuel + Bet365) via Odds API",
        [PY, "odds/fetch_points_props_oddsapi.py"],
    ),
    (
        "Normalize props",
        [PY, "odds/normalize_points_props.py"],
    ),
    (
        "Run inference (NBA Player Points Model)",
        [PY, "inference/predict_today_points_props_regression.py"],
    ),
]


def run_step(title: str, cmd: list[str]) -> None:
    print("\n" + "=" * 50)
    print(f"▶ {title}")
    print(f"$ {' '.join(cmd)}")
    print("=" * 50)

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("\n❌ PIPELINE STOPPED")
        print(f"Failed step: {title}")
        print(f"Command: {' '.join(cmd)}")
        sys.exit(1)


def main() -> None:
    # Quick check: show which python we're using
    print(f"\n🐍 Using Python: {sys.executable}")

    required_files = [
        Path("models/minutes_xgb.json"),
        Path("data/raw/player_game_logs.csv"),
        Path("data/raw/fetch_current_player_teams.py"),
        Path("odds/fetch_points_props_oddsapi.py"),
        Path("odds/normalize_points_props.py"),
        Path("inference/predict_today_points_props_regression.py"),
    ]
    for f in required_files:
        if not f.exists():
            print(f"\n❌ Missing required file: {f}")
            sys.exit(1)

    print("\n🚀 Running NBA Player Points Model pipeline...\n")
    for title, cmd in STEPS:
        run_step(title, cmd)

    print("\n" + "=" * 50)
    print("✅ PIPELINE COMPLETE")
    print("📄 Output file:")
    print("   data/processed/today_points_prop_predictions.xlsx")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
