from __future__ import annotations
import pandas as pd
from pathlib import Path

NBA_LOGS = Path("data/raw/player_game_logs.csv")
PROPS_NORM = Path("data/odds_logs/points_props_normalized.csv")
OUT_PATH = Path("data/processed/points_props_training.csv")

def norm_name(s: str) -> str:
    s = str(s).lower().strip()
    for tok in [" jr.", " sr.", " iii", " ii", " iv"]:
        s = s.replace(tok, "")
    s = s.replace(".", "").replace("'", "")
    s = " ".join(s.split())
    return s

def norm_team(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace(".", "")
    s = " ".join(s.split())
    return s

def main():
    nba = pd.read_csv(NBA_LOGS)
    nba["GAME_DATE_STR"] = pd.to_datetime(nba["GAME_DATE"]).dt.date.astype(str)
    nba["player_norm"] = nba["PLAYER_NAME"].apply(norm_name)
    nba["team_norm"] = nba["TEAM_NAME"].apply(norm_team)

    props = pd.read_csv(PROPS_NORM)
    props["commence_time"] = pd.to_datetime(props["commence_time"], utc=True, errors="coerce")
    props["GAME_DATE_STR"] = props["commence_time"].dt.date.astype(str)
    props["player_norm"] = props["player"].apply(norm_name)

    # Expand: one row per (prop, team) because we don't know which team the player is on from props alone
    props_home = props.copy()
    props_home["TEAM_NAME"] = props_home["home_team"]
    props_away = props.copy()
    props_away["TEAM_NAME"] = props_away["away_team"]

    props_expanded = pd.concat([props_home, props_away], ignore_index=True)
    props_expanded["team_norm"] = props_expanded["TEAM_NAME"].apply(norm_team)

    # Merge on date + team + player name
    merged = props_expanded.merge(
        nba,
        left_on=["GAME_DATE_STR", "team_norm", "player_norm"],
        right_on=["GAME_DATE_STR", "team_norm", "player_norm"],
        how="inner"
    )

    if merged.empty:
        print("No rows matched.")
        print("Sample props teams:", props_expanded["TEAM_NAME"].dropna().unique()[:10])
        print("Sample nba teams:", nba["TEAM_NAME"].dropna().unique()[:10])
        print("Sample props players:", props_expanded["player"].dropna().unique()[:10])
        print("Sample nba players:", nba["PLAYER_NAME"].dropna().unique()[:10])
        return

    merged["y_over"] = (merged["PTS"] > merged["line"]).astype(int)

    out_cols = [
        "event_id", "commence_time", "GAME_DATE_STR",
        "home_team", "away_team",
        "book_key", "book_title",
        "player", "line", "odds_over", "odds_under",
        "p_over_implied", "p_under_implied",
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "GAME_ID",
        "PTS", "y_over"
    ]

    out = merged[out_cols].drop_duplicates()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH.resolve()}")
    print("Rows matched:", len(out))
    print("Over rate:", out["y_over"].mean())
    print(out.head(10))

if __name__ == "__main__":
    main()
