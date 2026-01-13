from __future__ import annotations

import pandas as pd
from pathlib import Path

IN_DIR = Path("data/odds_logs")
OUT_PATH = IN_DIR / "assists_props_normalized.csv"

BOOK_PRIORITY = ["fanduel", "bet365"]


def main():
    files = sorted(IN_DIR.glob("assists_props_*.csv"))
    files = [f for f in files if "normalized" not in f.name.lower() and f.stat().st_size > 50]
    if not files:
        raise RuntimeError("No non-empty RAW assists logs found. Run fetch_assists_props_oddsapi.py first.")

    latest = max(files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest)

    required = {"commence_time", "event_id", "home_team", "away_team", "book_key", "book_title",
                "player", "side", "line", "odds"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"RAW assists log missing columns: {missing}. File: {latest} | cols={list(df.columns)}")

    df["side"] = df["side"].astype(str).str.lower().str.strip()
    df["book_key"] = df["book_key"].astype(str).str.lower().str.strip()

    # keep priority books only
    df = df[df["book_key"].isin(BOOK_PRIORITY)].copy()
    if df.empty:
        raise RuntimeError("No rows after filtering to FanDuel/bet365. Assists props likely not posted yet.")

    df = df[df["side"].isin(["over", "under"])].copy()
    if df.empty:
        raise RuntimeError("No over/under rows found in RAW assists log.")

    # Choose best book per prop row (FD > bet365)
    df["book_rank"] = df["book_key"].apply(lambda b: BOOK_PRIORITY.index(b))
    df = df.sort_values(["commence_time", "event_id", "player", "line", "book_rank"])
    df = df.drop_duplicates(subset=["event_id", "player", "line", "side"], keep="first")

    pivot = (
        df.pivot_table(
            index=["commence_time", "event_id", "home_team", "away_team",
                   "book_key", "book_title", "player", "line"],
            columns="side",
            values="odds",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"over": "odds_over", "under": "odds_under"})
        .dropna(subset=["odds_over", "odds_under"])
    )

    pivot.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved normalized assists props: {OUT_PATH.resolve()} | rows={len(pivot)}")
    print(f"Used RAW file: {latest}")


if __name__ == "__main__":
    main()
