import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/processed/train_points.csv"
MODEL_PATH = "models/points_xgb.json"

FEATURES = [
    "min_last5", "min_last10",
    "pts_last5", "pts_last10",
    "fga_last5", "fga_last10",
    "fta_last10", "fg3a_last10",
    "tov_last10", "reb_last10",
    "pts_per_min_last10",
    "fga_per_min_last10", "fta_per_min_last10", "fg3a_per_min_last10",
    "pts_std_last10",
    "rest_days", "is_home",
    "opp_pts_allowed_last10",
    "opp_fga_allowed_last10",
    "opp_fta_allowed_last10",
    "opp_fg3a_allowed_last10",
]

def season_start_year(season: str) -> int:
    # season like "2024-25" -> 2024
    try:
        return int(str(season).split("-")[0])
    except Exception:
        return -1

def main():
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    seasons = sorted(df["SEASON"].dropna().unique(), key=season_start_year)
    if len(seasons) < 2:
        raise RuntimeError(f"Need at least 2 seasons, found: {seasons}")

    TRAIN_SEASON = seasons[-2]
    VAL_SEASON = seasons[-1]

    train_df = df[df["SEASON"] == TRAIN_SEASON].copy()
    val_df = df[df["SEASON"] == VAL_SEASON].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(
            f"Season split failed. Train rows: {len(train_df)}, Val rows: {len(val_df)}. "
            "Make sure you have both seasons."
        )

    X_train = train_df[FEATURES]
    y_train = train_df["delta_pts"]

    X_val = val_df[FEATURES]
    y_val = val_df["delta_pts"]

    # Baseline for delta is always 0 (i.e., predict pts_last10 with no adjustment)
    baseline_preds = [0.0] * len(y_val)
    baseline_mae = mean_absolute_error(y_val, baseline_preds)
    print(f"Baseline MAE (predict delta=0) on {VAL_SEASON}: {baseline_mae:.3f}")

    model = xgb.XGBRegressor(
        n_estimators=2500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    model_mae = mean_absolute_error(y_val, preds)

    print(f"XGBoost MAE (delta) on {VAL_SEASON}: {model_mae:.3f}")
    print(f"Improvement: {baseline_mae - model_mae:.3f} points")

    model.get_booster().save_model(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
