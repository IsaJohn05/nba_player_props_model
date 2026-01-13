from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SPORTS_ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing ODDS_API_KEY in your .env file")

OUT_DIR = Path("data/odds_logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Odds API settings (you may need to adjust these depending on The Odds API docs/plan)
SPORT = "basketball_nba"
REGIONS = "us"
MARKETS = "player_points"  # sometimes called "player_points" or similar
ODDS_FORMAT = "american"

BASE_URL = "https://api.the-odds-api.com/v4"


def fetch_events() -> list[dict]:
    url = f"{BASE_URL}/sports/{SPORT}/events"
    params = {"apiKey": API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_event_props(event_id: str) -> dict:
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def flatten_props(event_payload: dict, fetched_at_iso: str) -> list[dict]:
    rows = []
    event_id = event_payload.get("id")
    commence_time = event_payload.get("commence_time")
    home_team = event_payload.get("home_team")
    away_team = event_payload.get("away_team")

    for book in event_payload.get("bookmakers", []):
        book_key = book.get("key")
        book_title = book.get("title")
        last_update = book.get("last_update")

        for market in book.get("markets", []):
            market_key = market.get("key")

            for outcome in market.get("outcomes", []):
                # For player props, outcomes often have:
                # name (player), description (Over/Under), point (line), price (odds)
                rows.append({
                    "fetched_at": fetched_at_iso,
                    "event_id": event_id,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book_key": book_key,
                    "book_title": book_title,
                    "book_last_update": last_update,
                    "market": market_key,
                    "player": outcome.get("description"),
                    "side": outcome.get("name"),
                    "line": outcome.get("point"),
                    "odds": outcome.get("price"),
                })

    return rows


def main():
    fetched_at_iso = datetime.now(timezone.utc).isoformat()

    events = fetch_events()
    print(f"Found {len(events)} NBA events")

    all_rows = []
    for i, ev in enumerate(events, start=1):
        event_id = ev.get("id")
        if not event_id:
            continue

        try:
            payload = fetch_event_props(event_id)
            all_rows.extend(flatten_props(payload, fetched_at_iso))
            print(f"[{i}/{len(events)}] grabbed props for event {event_id}")
        except Exception as e:
            print(f"[{i}/{len(events)}] failed event {event_id}: {e}")

        time.sleep(0.4)  # be nice to the API

    if not all_rows:
        print("No prop rows returned. You may need to adjust MARKETS for your plan.")
        return

    df = pd.DataFrame(all_rows)

    out_path = OUT_DIR / f"points_props_{fetched_at_iso.replace(':','-')}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path.resolve()}")
    print(df.head(10))


if __name__ == "__main__":
    main()
