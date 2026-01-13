from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH = Path("data/raw/player_game_logs.csv")
OUT_PATH = Path("data/processed/train_minutes.csv")

def main():
    df = pd.read_csv(IN_PATH)

    # Basic cleanup
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # Some nba logs have START_POSITION; if not, we’ll create a safe starter proxy
    if "START_POSITION" in df.columns:
        df["is_starter"] = (df["START_POSITION"].astype(str).str.strip() != "").astype(int)
    else:
        # Proxy: starter-ish if MIN >= 24 (not perfect, but usable)
        df["is_starter"] = (pd.to_numeric(df["MIN"], errors="coerce") >= 24).astype(int)

    # Ensure numeric columns
    for c in ["MIN", "PTS", "FGA", "FTA", "FG3A", "TOV", "REB"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rolling features (shift(1) to avoid leakage)
    g = df.groupby("PLAYER_ID", group_keys=False)

    df["min_last5"] = g["MIN"].apply(lambda s: s.shift(1).rolling(5).mean())
    df["min_last10"] = g["MIN"].apply(lambda s: s.shift(1).rolling(10).mean())
    df["min_std_last10"] = g["MIN"].apply(lambda s: s.shift(1).rolling(10).std())

    df["pts_last10"] = g["PTS"].apply(lambda s: s.shift(1).rolling(10).mean())
    df["fga_last10"] = g["FGA"].apply(lambda s: s.shift(1).rolling(10).mean())
    df["fta_last10"] = g["FTA"].apply(lambda s: s.shift(1).rolling(10).mean())
    df["fg3a_last10"] = g["FG3A"].apply(lambda s: s.shift(1).rolling(10).mean())
    df["tov_last10"] = g["TOV"].apply(lambda s: s.shift(1).rolling(10).mean())
    df["reb_last10"] = g["REB"].apply(lambda s: s.shift(1).rolling(10).mean())

    # Rest days + home/away
    df["rest_days"] = g["GAME_DATE"].apply(lambda s: (s - s.shift(1)).dt.days)
    df["is_b2b"] = (df["rest_days"] == 1).astype(int)

    df["is_home"] = df["MATCHUP"].astype(str).str.contains(" vs. ").astype(int)

    # Target
    df["MIN_TARGET"] = df["MIN"]

    # Keep rows where features exist
    feature_cols = [
        "min_last5", "min_last10", "min_std_last10",
        "pts_last10", "fga_last10", "fta_last10", "fg3a_last10",
        "tov_last10", "reb_last10",
        "rest_days", "is_b2b", "is_home",
        "is_starter",
    ]

    keep_cols = ["PLAYER_ID", "PLAYER_NAME", "GAME_DATE"] + feature_cols + ["MIN_TARGET"]

    out = df[keep_cols].dropna(subset=feature_cols + ["MIN_TARGET"]).copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH.resolve()}")
    print("Rows:", len(out))
    print("Columns:", out.columns.tolist())

if __name__ == "__main__":
    main()
