from __future__ import annotations
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/processed/train_minutes.csv")
MODEL_PATH = Path("models/minutes_xgb.json")

TRAIN_SEASON_START = "2022-07-01"
VAL_SEASON_START = "2023-07-01"

FEATURES = [
    "min_last5", "min_last10", "min_std_last10",
    "pts_last10", "fga_last10", "fta_last10", "fg3a_last10",
    "tov_last10", "reb_last10",
    "rest_days", "is_b2b", "is_home",
    "is_starter",
]

def main():
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Time split (simple + safe)
    train_df = df[df["GAME_DATE"] < pd.to_datetime(VAL_SEASON_START)].copy()
    val_df = df[df["GAME_DATE"] >= pd.to_datetime(VAL_SEASON_START)].copy()

    X_train, y_train = train_df[FEATURES], train_df["MIN_TARGET"]
    X_val, y_val = val_df[FEATURES], val_df["MIN_TARGET"]

    # Baseline: predict min_last10
    baseline_pred = X_val["min_last10"].values
    baseline_mae = mean_absolute_error(y_val, baseline_pred)
    print(f"Baseline MAE (min_last10) on val: {baseline_mae:.3f}")

    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    print(f"XGBoost MAE (minutes) on val: {mae:.3f}")
    print(f"Improvement: {baseline_mae - mae:+.3f} minutes")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(MODEL_PATH))
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
