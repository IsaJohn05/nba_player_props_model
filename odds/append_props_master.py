from __future__ import annotations
from pathlib import Path
import pandas as pd

NEW_PATH = Path("data/odds_logs/points_props_normalized.csv")
MASTER_PATH = Path("data/odds_logs/points_props_master.csv")

def main():
    new = pd.read_csv(NEW_PATH)
    new["commence_time"] = pd.to_datetime(new["commence_time"], utc=True, errors="coerce")

    if MASTER_PATH.exists():
        master = pd.read_csv(MASTER_PATH)
        master["commence_time"] = pd.to_datetime(master["commence_time"], utc=True, errors="coerce")
        combined = pd.concat([master, new], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["event_id", "book_key", "player", "line", "commence_time"],
            keep="last"
        )
    else:
        combined = new

    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(MASTER_PATH, index=False)
    print(f"Master saved: {MASTER_PATH.resolve()}")
    print("Rows in master:", len(combined))

if __name__ == "__main__":
    main()
