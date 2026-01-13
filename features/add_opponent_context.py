from __future__ import annotations
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/raw/player_game_logs.csv")
OUT_PATH = Path("data/processed/train_points.csv")

def main():
    # Load raw player logs
    df = pd.read_csv(IN_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # --- Build TEAM-GAME totals from player logs ---
    team_game = (
        df.groupby(["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE", "SEASON"], as_index=False)
          .agg(
              team_pts=("PTS", "sum"),
              team_fga=("FGA", "sum"),
              team_fta=("FTA", "sum"),
              team_fg3a=("FG3A", "sum"),
              team_tov=("TOV", "sum"),
          )
    )

    # --- Attach opponent totals by pairing the other team in the same GAME_ID ---
    # For each GAME_ID, there are two TEAM_ID rows. Merge to get opponent’s stats.
    opp = team_game.rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "TEAM_ABBREVIATION": "OPP_TEAM_ABBR",
        "team_pts": "opp_pts",
        "team_fga": "opp_fga",
        "team_fta": "opp_fta",
        "team_fg3a": "opp_fg3a",
        "team_tov": "opp_tov",
    })

    team_with_opp = team_game.merge(opp, on=["GAME_ID", "GAME_DATE", "SEASON"], how="inner")
    team_with_opp = team_with_opp[team_with_opp["TEAM_ID"] != team_with_opp["OPP_TEAM_ID"]]

    # Now team_with_opp has one row per team-game with opponent totals attached.

    # --- Compute opponent "allowed" rolling features (defense proxy) ---
    team_with_opp = team_with_opp.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # Opponent allowed points = opponent’s points scored vs this team, i.e. opp_pts
    # Rolling means per TEAM_ID: how many points this team allows recently
    team_with_opp["opp_pts_allowed_last10"] = (
        team_with_opp["opp_pts"].shift(1).rolling(10).mean()
    )
    team_with_opp["opp_fga_allowed_last10"] = (
        team_with_opp["opp_fga"].shift(1).rolling(10).mean()
    )
    team_with_opp["opp_fta_allowed_last10"] = (
        team_with_opp["opp_fta"].shift(1).rolling(10).mean()
    )
    team_with_opp["opp_fg3a_allowed_last10"] = (
        team_with_opp["opp_fg3a"].shift(1).rolling(10).mean()
    )

    # Keep only what we need to merge onto player rows
    team_def = team_with_opp[[
        "GAME_ID", "TEAM_ID", "SEASON",
        "opp_pts_allowed_last10",
        "opp_fga_allowed_last10",
        "opp_fta_allowed_last10",
        "opp_fg3a_allowed_last10",
    ]].copy()

    # --- Load your processed training set and merge opponent context in ---
    train = pd.read_csv(OUT_PATH)
    train["GAME_DATE"] = pd.to_datetime(train["GAME_DATE"])

    # We need GAME_ID and TEAM_ID in processed data—so we’ll rebuild processed with those.
    # For now, warn clearly if missing:
    missing = [c for c in ["GAME_ID", "TEAM_ID"] if c not in train.columns]
    if missing:
        raise RuntimeError(
            f"Your processed file is missing {missing}. "
            "We need to include GAME_ID and TEAM_ID in build_points_features.py final_cols."
        )

    # Merge opponent defense by matching opponent team id to the defense table's TEAM_ID
    train = train.merge(
        team_def,
        left_on=["GAME_ID", "OPP_TEAM_ID", "SEASON"],
        right_on=["GAME_ID", "TEAM_ID", "SEASON"],
        how="left",
        suffixes=("", "_drop")
    )
    # Clean up extra join key column from the right table
    if "TEAM_ID_drop" in train.columns:
        train = train.drop(columns=["TEAM_ID_drop"])

    # Drop rows without enough opponent history
    train = train.dropna(subset=[
        "opp_pts_allowed_last10",
        "opp_fga_allowed_last10",
        "opp_fta_allowed_last10",
        "opp_fg3a_allowed_last10",
    ])

    train.to_csv(OUT_PATH, index=False)
    print(f"Updated training file saved with opponent context: {OUT_PATH.resolve()}")
    print("New columns added:",
          ["opp_pts_allowed_last10", "opp_fga_allowed_last10", "opp_fta_allowed_last10", "opp_fg3a_allowed_last10"])

if __name__ == "__main__":
    main()
