from __future__ import annotations

import pandas as pd
from pathlib import Path

IN_DIR = Path("data/odds_logs")
OUT_PATH = IN_DIR / "rebounds_props_normalized.csv"

BOOK_PRIORITY = ["fanduel", "bet365"]


def main():
    files = sorted(IN_DIR.glob("rebounds_props_*.csv"))
    files = [f for f in files if "normalized" not in f.name.lower()]
    if not files:
        raise RuntimeError("No raw rebounds props logs found.")

    # Read all non-empty files and concatenate
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            if not df.empty and len(df.columns) > 0:
                dfs.append(df)
        except pd.errors.EmptyDataError:
            continue
    
    if not dfs:
        raise RuntimeError("No non-empty rebounds props files found.")
    
    df = pd.concat(dfs, ignore_index=True)

    def _infer_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    side_col = _infer_col(df, ["side", "name", "outcome", "outcome_name", "selection", "label"])
    line_col = _infer_col(df, ["line", "point", "handicap"])
    odds_col = _infer_col(df, ["odds", "price", "american_odds"])
    book_key_col = _infer_col(df, ["book_key", "site_key", "book", "provider", "market"])

    missing = []
    if side_col is None:
        missing.append("side")
    if line_col is None:
        missing.append("line")
    if odds_col is None:
        missing.append("odds")
    if book_key_col is None:
        missing.append("book_key")
    if missing:
        raise RuntimeError(
            f"Could not infer columns {missing} from rebounds props. Columns found: {list(df.columns)}"
        )

    # Standardize into expected names
    df["side"] = df[side_col].astype(str).str.lower().str.strip()
    df["line"] = df[line_col]
    df["odds"] = df[odds_col]
    df["book_key"] = df[book_key_col].astype(str).str.lower().str.strip()

    # Keep desired books
    df = df[df["book_key"].isin(BOOK_PRIORITY)].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering to FanDuel/bet365.")

    df["book_rank"] = df["book_key"].apply(lambda x: BOOK_PRIORITY.index(x))
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
    )

    pivot = pivot.rename(columns={"over": "odds_over", "under": "odds_under"})
    pivot = pivot.dropna(subset=["odds_over", "odds_under"])

    pivot.to_csv(OUT_PATH, index=False)
    print(f"Saved normalized rebounds props: {OUT_PATH.resolve()} | rows={len(pivot)}")


if __name__ == "__main__":
    main()
