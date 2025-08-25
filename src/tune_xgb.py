# -*- coding: utf-8 -*-
"""
Randomized hyperparameter search for XGBoost.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from joblib import dump

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target_col", default="temperature_2m")
    p.add_argument("--save_path", default="models/xgb_tuned.pkl")
    p.add_argument("--n_iter", type=int, default=30)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    y = df[args.target_col]
    drop_cols = [args.target_col]
    if "time" in df.columns:
        drop_cols.append("time")
    X = df.drop(columns=drop_cols)

    split_idx = int(len(df)*0.8)
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(tree_method="hist", n_jobs=-1, random_state=args.random_state)

    param_dist = {
        "n_estimators": np.arange(400, 3001, 200),
        "max_depth": np.arange(4, 13),
        "learning_rate": np.linspace(0.01, 0.2, 20),
        "subsample": np.linspace(0.6, 1.0, 9),
        "colsample_bytree": np.linspace(0.6, 1.0, 9),
        "reg_alpha": np.linspace(0.0, 0.5, 6),
        "reg_lambda": np.linspace(0.5, 2.0, 16)
    }

    scorer = make_scorer(rmse, greater_is_better=False)
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring=scorer,
        random_state=args.random_state,
        n_jobs=-1,
        cv=3,
        verbose=1
    )
    rs.fit(X_train, y_train)

    best_model = rs.best_estimator_
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_valid)
    mae = float(mean_absolute_error(y_valid, preds))
    try:
        rmse = float(mean_squared_error(y_valid, preds, squared=False))
    except TypeError:
        rmse = float(mean_squared_error(y_valid, preds)) ** 0.5

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    dump(best_model, args.save_path)
    with open("models/xgb_tuned_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_params": rs.best_params_
        }, f, ensure_ascii=False, indent=2)

    print("Best params:", rs.best_params_)
    print(f"MAE={mae:.3f} | RMSE={rmse_val:.3f} (validation set)")
    print(f"Saved tuned model to {args.save_path} and report to models/xgb_tuned_report.json")

if __name__ == "__main__":
    main()