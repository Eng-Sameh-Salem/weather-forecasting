# -*- coding: utf-8 -*-
"""
Train an XGBoost regressor for temperature forecasting.
- Chronological 80/20 split
- Early stopping on the validation set
- Saves model and metrics
"""
import argparse, json, os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from joblib import dump

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV created by make_features.py")
    p.add_argument("--target_col", default="temperature_2m")
    p.add_argument("--save_path", default="models/xgb_temp.pkl")
    p.add_argument("--metrics_path", default="models/metrics_xgb.json")
    p.add_argument("--n_estimators", type=int, default=2000)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
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
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        tree_method="hist",
        n_jobs=-1
    )

        # training with graceful fallback if early_stopping_rounds is unsupported
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=args.early_stopping_rounds
        )
    except TypeError:
        # Older xgboost versions: no early_stopping_rounds in fit
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        print("Warning: early stopping disabled (xgboost version doesn't support 'early_stopping_rounds' in .fit).")


    preds = model.predict(X_valid)
    mae = float(mean_absolute_error(y_valid, preds))
    try:
        rmse = float(mean_squared_error(y_valid, preds, squared=False))
    except TypeError:
        rmse = float(mean_squared_error(y_valid, preds)) ** 0.5

    dump(model, args.save_path)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump({"MAE": mae, "RMSE": rmse, "n_valid": int(len(y_valid))}, f, ensure_ascii=False, indent=2)

    print(f"Saved XGBoost model to: {args.save_path}")
    print(f"MAE={mae:.3f} | RMSE={rmse:.3f} | valid_samples={len(y_valid)}")

if __name__ == "__main__":
    main()