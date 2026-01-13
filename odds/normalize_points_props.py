from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd

IN_DIR = Path("data/odds_logs")
OUT_PATH = Path("data/odds_logs/points_props_normalized.csv")


def american_to_implied(odds):
    if pd.isna(odds):
        return float("nan")
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return (-odds) / ((-odds) + 100)


def main():
    files = sorted(glob.glob(str(IN_DIR / "points_props_*.csv")))
    if not files:
        raise RuntimeError("No prop files found")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Fix swapped columns
    df["player"] = df["player"].astype(str).str.strip()
    df["side"] = df["side"].astype(str).str.strip()

    swapped = df["player"].str.lower().isin(["over", "under"])
    df.loc[swapped, ["player", "side"]] = df.loc[swapped, ["side", "player"]].values

    # Normalize
    df["side"] = df["side"].str.lower()
    df = df[df["side"].isin(["over", "under"])]

    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)

    key_cols = [
        "event_id", "commence_time",
        "home_team", "away_team",
        "book_key", "book_title",
        "player", "line"
    ]

    pivot = (
        df.pivot_table(
            index=key_cols,
            columns="side",
            values="odds",
            aggfunc="first"
        )
        .reset_index()
    )

    pivot = pivot.rename(columns={
        "over": "odds_over",
        "under": "odds_under"
    })

    pivot["p_over_implied"] = pivot["odds_over"].apply(american_to_implied)
    pivot["p_under_implied"] = pivot["odds_under"].apply(american_to_implied)
    pivot["game_date_utc"] = pivot["commence_time"].dt.date.astype(str)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUT_PATH, index=False)

    print("✅ Normalized props saved")
    print(pivot[["player", "line", "odds_over", "odds_under"]].head())
    print("Rows:", len(pivot))


if __name__ == "__main__":
    main()
