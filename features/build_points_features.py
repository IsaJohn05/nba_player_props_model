from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/player_game_logs.csv")
OUT_PATH = Path("data/processed/train_points.csv")

ROLLING_WINDOWS = [5, 10]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    feature_rows = []

    for _, pdf in df.groupby("PLAYER_ID"):
        pdf = pdf.sort_values("GAME_DATE")

        # Base rolling means (shift(1) ensures we only use past games)
        for window in ROLLING_WINDOWS:
            pdf[f"min_last{window}"] = pdf["MIN"].shift(1).rolling(window).mean()
            pdf[f"pts_last{window}"] = pdf["PTS"].shift(1).rolling(window).mean()
            pdf[f"fga_last{window}"] = pdf["FGA"].shift(1).rolling(window).mean()

        # More rolling means (usage/role proxies)
        pdf["fta_last10"] = pdf["FTA"].shift(1).rolling(10).mean()
        pdf["fg3a_last10"] = pdf["FG3A"].shift(1).rolling(10).mean()
        pdf["tov_last10"] = pdf["TOV"].shift(1).rolling(10).mean()
        pdf["reb_last10"] = pdf["REB"].shift(1).rolling(10).mean()

        # Efficiency / involvement proxy
        pdf["pts_per_min_last10"] = (
            pdf["PTS"].shift(1).rolling(10).sum()
            / pdf["MIN"].shift(1).rolling(10).sum()
        )

        # Per-minute volume rates
        pdf["fga_per_min_last10"] = (
            pdf["FGA"].shift(1).rolling(10).sum()
            / pdf["MIN"].shift(1).rolling(10).sum()
        )
        pdf["fta_per_min_last10"] = (
            pdf["FTA"].shift(1).rolling(10).sum()
            / pdf["MIN"].shift(1).rolling(10).sum()
        )
        pdf["fg3a_per_min_last10"] = (
            pdf["FG3A"].shift(1).rolling(10).sum()
            / pdf["MIN"].shift(1).rolling(10).sum()
        )

        # Volatility
        pdf["pts_std_last10"] = pdf["PTS"].shift(1).rolling(10).std()

        # Rest days
        pdf["rest_days"] = (pdf["GAME_DATE"] - pdf["GAME_DATE"].shift(1)).dt.days

        # Home / away
        pdf["is_home"] = pdf["MATCHUP"].str.contains(" vs. ").astype(int)

        feature_rows.append(pdf)

    df_feat = pd.concat(feature_rows, ignore_index=True)

    required_cols = [
        "min_last5", "min_last10",
        "pts_last5", "pts_last10",
        "fga_last5", "fga_last10",
        "fta_last10", "fg3a_last10",
        "tov_last10", "reb_last10",
        "pts_per_min_last10",
        "fga_per_min_last10", "fta_per_min_last10", "fg3a_per_min_last10",
        "pts_std_last10",
        "rest_days"
    ]

    df_feat = df_feat.dropna(subset=required_cols)

    # ✅ NEW TARGET: residual vs baseline
    # delta_pts = actual points - rolling baseline (pts_last10)
    df_feat["delta_pts"] = df_feat["PTS"] - df_feat["pts_last10"]

    final_cols = [
        "SEASON",
        "GAME_ID", "TEAM_ID",
        "PLAYER_ID", "PLAYER_NAME", "GAME_DATE",

        "min_last5", "min_last10",
        "pts_last5", "pts_last10",
        "fga_last5", "fga_last10",
        "fta_last10", "fg3a_last10",
        "tov_last10", "reb_last10",
        "pts_per_min_last10",
        "fga_per_min_last10", "fta_per_min_last10", "fg3a_per_min_last10",
        "pts_std_last10",
        "rest_days", "is_home",

        # label
        "delta_pts"
    ]

    return df_feat[final_cols].reset_index(drop=True)


def main():
    df = pd.read_csv(RAW_PATH)
    df_final = build_features(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH.resolve()}")
    print(f"Rows: {len(df_final):,}")
    print("Columns:", list(df_final.columns))


if __name__ == "__main__":
    main()
