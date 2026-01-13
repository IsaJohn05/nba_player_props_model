from pathlib import Path
import pandas as pd

PROPS_PATH = Path("data/odds_logs/points_props_master.csv")
GAMES_PATH = Path("data/raw/player_game_logs_recent.csv")
OUT_PATH = Path("data/processed/points_props_labeled.csv")

BOOK_PRIORITY = ["fanduel", "bet365"]

def norm(s):
    return (
        str(s).lower()
        .replace(".", "")
        .replace("'", "")
        .replace(" jr", "")
        .replace(" sr", "")
        .strip()
    )

def main():
    props = pd.read_csv(PROPS_PATH)
    games = pd.read_csv(GAMES_PATH)

    props["player_norm"] = props["player"].apply(norm)
    games["player_norm"] = games["PLAYER_NAME"].apply(norm)

    props["commence_time"] = pd.to_datetime(props["commence_time"], utc=True)
    props["GAME_DATE"] = props["commence_time"].dt.date.astype(str)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date.astype(str)

    # keep only FD + bet365
    props = props[props["book_key"].isin(BOOK_PRIORITY)]

    # choose best book per (event, player)
    props["book_rank"] = props["book_key"].apply(lambda b: BOOK_PRIORITY.index(b))
    props = props.sort_values("book_rank")
    props = props.drop_duplicates(
        subset=["event_id", "player", "line", "GAME_DATE"],
        keep="first"
    )

    merged = props.merge(
        games,
        on=["GAME_DATE", "player_norm"],
        how="inner"
    )

    merged["y_over"] = (merged["PTS"] > merged["line"]).astype(int)

    keep_cols = [
        "GAME_DATE", "event_id",
        "player", "PLAYER_ID", "TEAM_NAME",
        "book_key", "line",
        "odds_over", "odds_under",
        "p_over_implied", "p_under_implied",
        "PTS", "y_over"
    ]

    out = merged[keep_cols]

    if OUT_PATH.exists():
        old = pd.read_csv(OUT_PATH)
        out = pd.concat([old, out], ignore_index=True).drop_duplicates()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Labeled props saved: {OUT_PATH}")
    print("Rows:", len(out))
    print("Over rate:", out["y_over"].mean())

if __name__ == "__main__":
    main()
