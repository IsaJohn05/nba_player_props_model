from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

API_KEY = os.environ.get("SPORTS_ODDS_API_KEY")
SPORT = "basketball_nba"
REGIONS = "us"
MARKETS = "player_rebounds"
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"

BOOKS = ["fanduel", "bet365"]

OUT_DIR = Path("data/odds_logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4"


def main():
    if not API_KEY:
        raise RuntimeError("Missing API key.")

    events = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events",
        params={"apiKey": API_KEY, "dateFormat": DATE_FORMAT},
    ).json()

    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    print(f"Found {len(events)} NBA events")

    for i, ev in enumerate(events, start=1):
        odds = requests.get(
            f"{BASE_URL}/sports/{SPORT}/events/{ev['id']}/odds",
            params={
                "apiKey": API_KEY,
                "regions": REGIONS,
                "markets": MARKETS,
                "oddsFormat": ODDS_FORMAT,
                "dateFormat": DATE_FORMAT,
            },
        ).json()

        for b in odds.get("bookmakers", []):
            if b["key"] not in BOOKS:
                continue
            for m in b.get("markets", []):
                if m["key"] != MARKETS:
                    continue
                for o in m.get("outcomes", []):
                    rows.append({
                        "fetched_at": fetched_at,
                        "event_id": ev["id"],
                        "commence_time": ev["commence_time"],
                        "home_team": ev["home_team"],
                        "away_team": ev["away_team"],
                        "book_key": b["key"],
                        "book_title": b["title"],
                        "player": o.get("description"),
                        "side": o["name"],
                        "line": o["point"],
                        "odds": o["price"],
                    })

        print(f"[{i}/{len(events)}] grabbed rebounds props")

    df = pd.DataFrame(rows)
    out = OUT_DIR / f"rebounds_props_{fetched_at.replace(':','-')}.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
