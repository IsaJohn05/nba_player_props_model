from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

# -----------------------
# Paths
# -----------------------
NBA_LOGS = Path("data/raw/player_game_logs.csv")  # historical player games
PROPS_NORM = Path("data/odds_logs/points_props_normalized.csv")  # today's props (normalized)
MODEL_PATH = Path("models/points_over_xgb.json")  # your classifier booster
OUT_PATH = Path("data/processed/today_points_prop_predictions.csv")

BOOK_PRIORITY = ["fanduel", "bet365"]


# -----------------------
# Helpers
# -----------------------
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


def american_to_implied(odds) -> float:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return float("nan")
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return (-odds) / ((-odds) + 100.0)


# -----------------------
# Feature builders
# -----------------------
def build_latest_player_features(nba: pd.DataFrame) -> pd.DataFrame:
    """
    Builds one row per player using ONLY past games (rolling features).
    Output indexed by player_norm.
    """
    nba = nba.copy()
    nba["GAME_DATE"] = pd.to_datetime(nba["GAME_DATE"])
    nba = nba.sort_values(["PLAYER_ID", "GAME_DATE"])

    # rolling features per player
    feats = []
    for _, pdf in nba.groupby("PLAYER_ID"):
        pdf = pdf.sort_values("GAME_DATE")

        # shift(1) so "latest feature row" corresponds to next game prediction
        pdf["min_last5"] = pdf["MIN"].shift(1).rolling(5).mean()
        pdf["min_last10"] = pdf["MIN"].shift(1).rolling(10).mean()

        pdf["pts_last5"] = pdf["PTS"].shift(1).rolling(5).mean()
        pdf["pts_last10"] = pdf["PTS"].shift(1).rolling(10).mean()

        pdf["fga_last5"] = pdf["FGA"].shift(1).rolling(5).mean()
        pdf["fga_last10"] = pdf["FGA"].shift(1).rolling(10).mean()

        pdf["fta_last10"] = pdf["FTA"].shift(1).rolling(10).mean()
        pdf["fg3a_last10"] = pdf["FG3A"].shift(1).rolling(10).mean()
        pdf["tov_last10"] = pdf["TOV"].shift(1).rolling(10).mean()
        pdf["reb_last10"] = pdf["REB"].shift(1).rolling(10).mean()

        pdf["pts_per_min_last10"] = (
            pdf["PTS"].shift(1).rolling(10).sum() / pdf["MIN"].shift(1).rolling(10).sum()
        )

        pdf["fga_per_min_last10"] = (
            pdf["FGA"].shift(1).rolling(10).sum() / pdf["MIN"].shift(1).rolling(10).sum()
        )
        pdf["fta_per_min_last10"] = (
            pdf["FTA"].shift(1).rolling(10).sum() / pdf["MIN"].shift(1).rolling(10).sum()
        )
        pdf["fg3a_per_min_last10"] = (
            pdf["FG3A"].shift(1).rolling(10).sum() / pdf["MIN"].shift(1).rolling(10).sum()
        )

        pdf["pts_std_last10"] = pdf["PTS"].shift(1).rolling(10).std()
        pdf["rest_days"] = (pdf["GAME_DATE"] - pdf["GAME_DATE"].shift(1)).dt.days

        pdf["is_home"] = pdf["MATCHUP"].astype(str).str.contains(" vs. ").astype(int)

        # take last available row (most recent game row) to represent player's current state
        last = pdf.iloc[-1:].copy()
        feats.append(last)

    f = pd.concat(feats, ignore_index=True)

    # add normalized name + last known team name/id (useful for matching)
    f["player_norm"] = f["PLAYER_NAME"].apply(norm_name)
    f["team_norm"] = f["TEAM_NAME"].apply(norm_team)

    # keep only columns we’ll need
    keep = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "player_norm", "team_norm",
        "min_last5", "min_last10",
        "pts_last5", "pts_last10",
        "fga_last5", "fga_last10",
        "fta_last10", "fg3a_last10",
        "tov_last10", "reb_last10",
        "pts_per_min_last10",
        "fga_per_min_last10", "fta_per_min_last10", "fg3a_per_min_last10",
        "pts_std_last10",
        "rest_days", "is_home",
    ]
    f = f[keep].drop_duplicates(subset=["player_norm"], keep="last")
    return f


def build_latest_team_defense(nba: pd.DataFrame) -> pd.DataFrame:
    """
    Builds one row per team: rolling 'allowed' metrics (last 10).
    We'll use TEAM_NAME norm as join key.
    """
    nba = nba.copy()
    nba["GAME_DATE"] = pd.to_datetime(nba["GAME_DATE"])

    # team-game totals from player logs
    team_game = (
        nba.groupby(["GAME_ID", "TEAM_ID", "TEAM_NAME", "GAME_DATE"], as_index=False)
           .agg(
               team_pts=("PTS", "sum"),
               team_fga=("FGA", "sum"),
               team_fta=("FTA", "sum"),
               team_fg3a=("FG3A", "sum"),
           )
    )

    # create opponent totals by self-merge on same GAME_ID
    opp = team_game.rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "TEAM_NAME": "OPP_TEAM_NAME",
        "team_pts": "opp_pts_scored",
        "team_fga": "opp_fga",
        "team_fta": "opp_fta",
        "team_fg3a": "opp_fg3a",
    })

    merged = team_game.merge(opp, on=["GAME_ID", "GAME_DATE"], how="inner")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()

    merged = merged.sort_values(["TEAM_ID", "GAME_DATE"])

    # rolling allowed (what opponents scored against this team)
    merged["opp_pts_allowed_last10"] = merged["opp_pts_scored"].shift(1).rolling(10).mean()
    merged["opp_fga_allowed_last10"] = merged["opp_fga"].shift(1).rolling(10).mean()
    merged["opp_fta_allowed_last10"] = merged["opp_fta"].shift(1).rolling(10).mean()
    merged["opp_fg3a_allowed_last10"] = merged["opp_fg3a"].shift(1).rolling(10).mean()

    # take latest row per team
    latest = merged.groupby("TEAM_ID").tail(1).copy()
    latest["team_norm"] = latest["TEAM_NAME"].apply(norm_team)

    keep = [
        "TEAM_ID", "TEAM_NAME", "team_norm",
        "opp_pts_allowed_last10", "opp_fga_allowed_last10",
        "opp_fta_allowed_last10", "opp_fg3a_allowed_last10"
    ]
    return latest[keep].drop_duplicates(subset=["team_norm"], keep="last")


# -----------------------
# Model loading + predict
# -----------------------
FEATURES = [
    "line",
    "min_last5", "min_last10",
    "pts_last5", "pts_last10",
    "fga_last5", "fga_last10",
    "fta_last10", "fg3a_last10",
    "tov_last10", "reb_last10",
    "pts_per_min_last10",
    "fga_per_min_last10", "fta_per_min_last10", "fg3a_per_min_last10",
    "pts_std_last10",
    "rest_days", "is_home",
    "opp_pts_allowed_last10", "opp_fga_allowed_last10",
    "opp_fta_allowed_last10", "opp_fg3a_allowed_last10",
]


def load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise RuntimeError(
            f"Missing model file: {path}. "
        )
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def main():
    nba = pd.read_csv(NBA_LOGS)
    props = pd.read_csv(PROPS_NORM)

    # filter to preferred books & choose best per (event, player, line)
    props["book_key"] = props["book_key"].astype(str).str.lower().str.strip()
    props = props[props["book_key"].isin(BOOK_PRIORITY)].copy()
    props["book_rank"] = props["book_key"].apply(lambda b: BOOK_PRIORITY.index(b))
    props = props.sort_values(["commence_time", "event_id", "player", "line", "book_rank"])
    props = props.drop_duplicates(subset=["event_id", "player", "line"], keep="first")

    # prep props
    props["player_norm"] = props["player"].apply(norm_name)
    props["home_norm"] = props["home_team"].apply(norm_team)
    props["away_norm"] = props["away_team"].apply(norm_team)

    # build features
    player_feats = build_latest_player_features(nba)
    team_def = build_latest_team_defense(nba)

    # merge props -> player features
    df = props.merge(player_feats, on="player_norm", how="left")

    # infer opponent team from today's matchup using player's last known team
    def get_opp_team_norm(row):
        t = row.get("team_norm")
        if pd.isna(t):
            return np.nan
        if t == row.get("home_norm"):
            return row.get("away_norm")
        if t == row.get("away_norm"):
            return row.get("home_norm")
        # if player team doesn't match either, we can't infer opp (trade / mismatch)
        return np.nan

    df["opp_team_norm"] = df.apply(get_opp_team_norm, axis=1)

    # merge opponent defense
    team_def_opp = team_def.rename(columns={"team_norm": "opp_team_norm"})
    df = df.merge(team_def_opp, on="opp_team_norm", how="left")

    # implied probs + edge
    df["p_over_implied"] = df.get("p_over_implied", np.nan)
    if "p_over_implied" not in df.columns or df["p_over_implied"].isna().all():
        df["p_over_implied"] = df["odds_over"].apply(american_to_implied)

    # drop rows missing required features
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns for prediction: {missing_cols}")

    pred_df = df.dropna(subset=FEATURES).copy()

    if pred_df.empty:
        print("No rows left after feature merge.")
        print("Common reasons:")
        print("- Player name mismatch between props and nba logs")
        print("- Player traded (team mismatch) so opponent can't be inferred")
        print("- Not enough game history for rolling features")
        return

    # load model + predict
    booster = load_booster(MODEL_PATH)
    dmat = xgb.DMatrix(pred_df[FEATURES].values, feature_names=FEATURES)
    p_over = booster.predict(dmat)

    pred_df["p_over_model"] = p_over
    pred_df["edge_over"] = pred_df["p_over_model"] - pred_df["p_over_implied"]

    # output columns
    out_cols = [
        "commence_time", "home_team", "away_team",
        "book_title", "player", "line",
        "odds_over", "odds_under",
        "p_over_implied", "p_over_model", "edge_over",
        "TEAM_NAME"
    ]
    out_cols = [c for c in out_cols if c in pred_df.columns]

    out = pred_df[out_cols].sort_values("edge_over", ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved predictions: {OUT_PATH.resolve()}")
    print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
