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
MARKETS = "player_assists"
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"

# Keep these as your priority books
BOOKS = ["fanduel", "bet365"]

OUT_DIR = Path("data/odds_logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4"


def main():
    if not API_KEY:
        raise RuntimeError("Missing API key. Set SPORTS_ODDS_API_KEY or ODDS_API_KEY.")


    events_resp = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events",
        params={"apiKey": API_KEY, "dateFormat": DATE_FORMAT},
        timeout=30,
    )
    if events_resp.status_code != 200:
        raise RuntimeError(f"Events request failed: {events_resp.status_code} {events_resp.text[:300]}")

    events = events_resp.json()
    print(f"Found {len(events)} NBA events")

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for i, ev in enumerate(events, start=1):
        odds_resp = requests.get(
            f"{BASE_URL}/sports/{SPORT}/events/{ev['id']}/odds",
            params={
                "apiKey": API_KEY,
                "regions": REGIONS,
                "markets": MARKETS,
                "oddsFormat": ODDS_FORMAT,
                "dateFormat": DATE_FORMAT,
                # IMPORTANT: ask API for these books (reduces noise)
                "bookmakers": ",".join(BOOKS),
            },
            timeout=30,
        )

        if odds_resp.status_code != 200:
            print(f"Odds request failed for event {ev['id']}: {odds_resp.status_code} {odds_resp.text[:200]}")
            continue

        odds = odds_resp.json()

        # Sometimes API returns {"message": "..."} instead of full payload
        if isinstance(odds, dict) and "message" in odds:
            print(f"API message for event {ev['id']}: {odds['message']}")
            continue

        bks = odds.get("bookmakers", []) if isinstance(odds, dict) else []
        market_outcomes = 0

        for b in bks:
            if b.get("key") not in BOOKS:
                continue
            for m in b.get("markets", []):
                if m.get("key") != MARKETS:
                    continue
                for o in m.get("outcomes", []):
                    # Odds API: outcomes have name=Over/Under, point=line, price=odds, description=player
                    rows.append({
                        "fetched_at": fetched_at,
                        "event_id": ev["id"],
                        "commence_time": ev.get("commence_time"),
                        "home_team": ev.get("home_team"),
                        "away_team": ev.get("away_team"),
                        "book_key": b.get("key"),
                        "book_title": b.get("title"),
                        "player": o.get("description"),
                        "side": o.get("name"),
                        "line": o.get("point"),
                        "odds": o.get("price"),
                    })
                    market_outcomes += 1

        print(f"[{i}/{len(events)}] assists outcomes for event {ev['id']}: {market_outcomes}")

    # DO NOT write empty files
    if len(rows) == 0:
        print("No assists props found for FanDuel/bet365.")
        print("This usually means assists props aren’t posted yet for those books. Try again later or expand BOOKS.")
        return

    df = pd.DataFrame(rows)
    out = OUT_DIR / f"assists_props_{fetched_at.replace(':','-')}.csv"
    df.to_csv(out, index=False)

    print(f"Saved: {out.resolve()} | rows={len(df)}")


if __name__ == "__main__":
    main()
