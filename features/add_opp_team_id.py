from __future__ import annotations
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/player_game_logs.csv")
PROC_PATH = Path("data/processed/train_points.csv")

def main():
    raw = pd.read_csv(RAW_PATH)
    raw = raw[["GAME_ID", "TEAM_ID"]].drop_duplicates()

    # For each game, map each TEAM_ID to the other TEAM_ID in that game
    raw["OPP_TEAM_ID"] = raw.groupby("GAME_ID")["TEAM_ID"].transform(lambda s: s.iloc[::-1].values)

    # Build mapping table (GAME_ID, TEAM_ID -> OPP_TEAM_ID)
    mapping = raw.drop_duplicates()

    proc = pd.read_csv(PROC_PATH)

    if "GAME_ID" not in proc.columns or "TEAM_ID" not in proc.columns:
        raise RuntimeError("Processed file must include GAME_ID and TEAM_ID. Rebuild features with those columns.")

    proc = proc.merge(mapping, on=["GAME_ID", "TEAM_ID"], how="left")

    if proc["OPP_TEAM_ID"].isna().any():
        # This should be rare; indicates games where the group wasn't exactly 2 teams
        print("Warning: Some OPP_TEAM_ID are missing. Dropping those rows.")
        proc = proc.dropna(subset=["OPP_TEAM_ID"])

    proc["OPP_TEAM_ID"] = proc["OPP_TEAM_ID"].astype(int)

    proc.to_csv(PROC_PATH, index=False)
    print("Added OPP_TEAM_ID to processed file:", PROC_PATH.resolve())
    print("Columns now include OPP_TEAM_ID:", "OPP_TEAM_ID" in proc.columns)

if __name__ == "__main__":
    main()
