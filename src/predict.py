# -*- coding: utf-8 -*-
"""
Generate multi-step forecasts from a trained model by rolling forward.
"""
import argparse
import pandas as pd
import numpy as np
from joblib import load

def build_features_for_step(hist_df, target_col="temperature_2m"):
    df = hist_df.copy()
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df["dayofweek"] = pd.to_datetime(df["time"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["time"]).dt.month

    for l in (1,2,3,6,12,24,48,72):
        df[f"{target_col}_lag{l}"] = df[target_col].shift(l)
    for w in (3,6,12,24,48,72):
        df[f"{target_col}_rollmean{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_rollstd{w}"] = df[target_col].rolling(w).std()

    row = df.dropna().iloc[-1].drop(labels=[target_col])
    return row

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recent_csv", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--target_col", default="temperature_2m")
    args = p.parse_args()

    model = load(args.model_path)
    hist = pd.read_csv(args.recent_csv)
    hist["time"] = pd.to_datetime(hist["time"])
    hist = hist.sort_values("time").reset_index(drop=True)

    preds = []
    last_time = hist["time"].iloc[-1]
    df_iter = hist.copy()

    for i in range(1, args.horizon+1):
        features = build_features_for_step(df_iter, target_col=args.target_col)
        X = features.drop(labels=["time"]) if "time" in features.index else features
        y_hat = model.predict([X.values])[0]
        new_row = {k: df_iter[k].iloc[-1] if k in df_iter.columns else np.nan for k in df_iter.columns}
        new_row[args.target_col] = y_hat
        new_row["time"] = last_time + pd.Timedelta(hours=i)
        df_iter = pd.concat([df_iter, pd.DataFrame([new_row])], ignore_index=True)
        preds.append({"time": new_row["time"], "prediction": float(y_hat)})

    out_df = pd.DataFrame(preds)
    out_path = "models/forecast.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Forecast saved to: {out_path}")
    print(out_df.tail())

if __name__ == "__main__":
    main()