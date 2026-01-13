from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PY = sys.executable  # ensures venv python is used


STEPS = [
    # --- shared ---
    ("Update current player → team map", [PY, "data/raw/fetch_current_player_teams.py"]),

    # --- POINTS ---
    ("Fetch today's POINTS props (FanDuel + Bet365)", [PY, "odds/fetch_points_props_oddsapi.py"]),
    ("Normalize POINTS props", [PY, "odds/normalize_points_props.py"]),
    ("Run POINTS inference (NBA Player Points Model)", [PY, "inference/predict_today_points_props_regression.py"]),

    # --- ASSISTS ---
    ("Fetch today's ASSISTS props (FanDuel + Bet365)", [PY, "odds/fetch_assists_props_oddsapi.py"]),
    ("Normalize ASSISTS props", [PY, "odds/normalize_assists_props.py"]),
    ("Run ASSISTS inference (NBA Player Assists Model)", [PY, "inference/predict_today_assists_props_regression.py"]),

    # --- REBOUNDS ---
    ("Fetch today's REBOUNDS props (FanDuel + Bet365)", [PY, "odds/fetch_rebounds_props_oddsapi.py"]),
    ("Normalize REBOUNDS props", [PY, "odds/normalize_rebounds_props.py"]),
    ("Run REBOUNDS inference (NBA Player Rebounds Model)", [PY, "inference/predict_today_rebounds_props_regression.py"]),
]


def run_step(title: str, cmd: list[str]) -> None:
    print("\n" + "=" * 60)
    print(f"{title}")
    print(f"$ {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("\nPIPELINE STOPPED")
        print(f"Failed step: {title}")
        print(f"Command: {' '.join(cmd)}")
        sys.exit(1)


def require(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        print(f"\nMissing required file: {p}")
        sys.exit(1)
    return p


def main() -> None:
    print(f"\nUsing Python: {sys.executable}")

    # --- Required shared files ---
    require("models/minutes_xgb.json")
    require("data/raw/player_game_logs.csv")
    require("data/raw/fetch_current_player_teams.py")

    # --- Required points ---
    require("odds/fetch_points_props_oddsapi.py")
    require("odds/normalize_points_props.py")
    require("inference/predict_today_points_props_regression.py")

    # --- Required assists ---
    require("odds/fetch_assists_props_oddsapi.py")
    require("odds/normalize_assists_props.py")
    require("inference/predict_today_assists_props_regression.py")

    # --- Required rebounds ---
    require("odds/fetch_rebounds_props_oddsapi.py")
    require("odds/normalize_rebounds_props.py")
    require("inference/predict_today_rebounds_props_regression.py")

    print("\nRunning NBA pipeline (Points + Assists + Rebounds)...\n")

    for title, cmd in STEPS:
        run_step(title, cmd)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("Output files:")
    print(" - data/processed/today_points_prop_predictions.xlsx")
    print(" - data/processed/today_assists_prop_predictions.xlsx")
    print(" - data/processed/today_rebounds_prop_predictions.xlsx")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
