from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

NBA_LOGS = Path("data/raw/player_game_logs.csv")
OUT_PATH = Path("data/processed/train_assists.csv")

def infer_season_from_date(d: pd.Timestamp) -> str:
    """
    NBA season label like '2023-24'.
    Season starts around Oct; if month >= 10 -> season is year-year+1 else year-1-year
    """
    y = d.year
    if d.month >= 10:
        return f"{y}-{str(y+1)[-2:]}"
    return f"{y-1}-{str(y)[-2:]}"

def main():
    if not NBA_LOGS.exists():
        raise RuntimeError(f"Missing NBA logs file: {NBA_LOGS}")

    nba = pd.read_csv(NBA_LOGS)

    # Required columns check
    required_cols = ["GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "MIN", "AST", "PTS",
                     "FGA", "FTA", "FG3A", "TOV", "REB", "MATCHUP"]
    missing = [c for c in required_cols if c not in nba.columns]
    if missing:
        raise RuntimeError(f"NBA logs missing required columns: {missing}")

    nba["GAME_DATE"] = pd.to_datetime(nba["GAME_DATE"])
    nba = nba.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()

    # numeric
    for c in ["MIN", "AST", "PTS", "FGA", "FTA", "FG3A", "TOV", "REB"]:
        nba[c] = pd.to_numeric(nba[c], errors="coerce")

    # Season label
    nba["SEASON"] = nba["GAME_DATE"].apply(infer_season_from_date)

    rows = []

    for pid, pdf in nba.groupby("PLAYER_ID"):
        pdf = pdf.sort_values("GAME_DATE").copy()

        # rolling inputs (shift(1) avoids leakage)
        pdf["min_last5"] = pdf["MIN"].shift(1).rolling(5).mean()
        pdf["min_last10"] = pdf["MIN"].shift(1).rolling(10).mean()
        pdf["min_std_last10"] = pdf["MIN"].shift(1).rolling(10).std()

        pdf["pts_last10"] = pdf["PTS"].shift(1).rolling(10).mean()
        pdf["fga_last10"] = pdf["FGA"].shift(1).rolling(10).mean()
        pdf["fta_last10"] = pdf["FTA"].shift(1).rolling(10).mean()
        pdf["fg3a_last10"] = pdf["FG3A"].shift(1).rolling(10).mean()
        pdf["tov_last10"] = pdf["TOV"].shift(1).rolling(10).mean()
        pdf["reb_last10"] = pdf["REB"].shift(1).rolling(10).mean()

        pdf["ast_last10"] = pdf["AST"].shift(1).rolling(10).mean()
        pdf["ast_per_min_last10"] = (
            pdf["AST"].shift(1).rolling(10).sum()
            / pdf["MIN"].shift(1).rolling(10).sum()
        )
        pdf["ast_std_last10"] = pdf["AST"].shift(1).rolling(10).std()

        # context
        pdf["rest_days"] = (pdf["GAME_DATE"] - pdf["GAME_DATE"].shift(1)).dt.days
        pdf["is_b2b"] = (pdf["rest_days"] == 1).astype(int)
        pdf["is_home"] = pdf["MATCHUP"].astype(str).str.contains(" vs. ").astype(int)

        if "START_POSITION" in pdf.columns:
            pdf["is_starter"] = (pdf["START_POSITION"].astype(str).str.strip() != "").astype(int)
        else:
            pdf["is_starter"] = (pdf["MIN"] >= 24).astype(int)

        # target: ast/min this game
        pdf["ast_per_min"] = pdf["AST"] / pdf["MIN"]

        # delta target (baseline = ast_per_min_last10)
        pdf["delta_ast_per_min"] = pdf["ast_per_min"] - pdf["ast_per_min_last10"]

        # filters
        keep = pdf.copy()
        keep = keep[keep["MIN"] >= 5]  # avoid tiny-minute games

        feature_cols = [
            "min_last5", "min_last10", "min_std_last10",
            "pts_last10", "fga_last10", "fta_last10", "fg3a_last10",
            "tov_last10", "reb_last10",
            "rest_days", "is_b2b", "is_home", "is_starter",
            "ast_last10", "ast_per_min_last10", "ast_std_last10",
        ]

        keep = keep.dropna(subset=feature_cols + ["delta_ast_per_min"]).copy()

        # Clamp extreme deltas (stabilizes training)
        keep["delta_ast_per_min"] = keep["delta_ast_per_min"].clip(lower=-0.25, upper=0.25)

        if not keep.empty:
            out_cols = [
                "SEASON", "GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "GAME_DATE",
                *feature_cols,
                "delta_ast_per_min",
            ]
            rows.append(keep[out_cols])

    if not rows:
        raise RuntimeError("No training rows created. Check if logs have enough games per player.")

    out = pd.concat(rows, ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH.resolve()}")
    print("Rows:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()
