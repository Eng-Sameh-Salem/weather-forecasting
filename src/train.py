# -*- coding: utf-8 -*-
"""
Train a RandomForest baseline on pre-built features.
Performs a simple chronological 80/20 split, saves the model and metrics.
"""
import argparse, json, os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target_col", default="temperature_2m")
    p.add_argument("--save_path", default="models/rf_temp.pkl")
    p.add_argument("--metrics_path", default="models/metrics.json")
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    df = pd.read_csv(args.data)
    y = df[args.target_col]
    drop_cols = [args.target_col]
    if "time" in df.columns:
        drop_cols.append("time")
    X = df.drop(columns=drop_cols)

    split_idx = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    try:
        rmse = float(mean_squared_error(y_test, preds, squared=False))
    except TypeError:
        rmse = float(mean_squared_error(y_test, preds)) ** 0.5

    dump(model, args.save_path)

    metrics = {"MAE": mae, "RMSE": rmse, "n_test": int(len(y_test))}
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {args.save_path}")
    print(f"MAE={mae:.3f} | RMSE={rmse:.3f} | test_samples={len(y_test)}")

if __name__ == "__main__":
    main()