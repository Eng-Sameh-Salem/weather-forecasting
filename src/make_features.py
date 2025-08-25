# -*- coding: utf-8 -*-
"""
Create time-series features:
- calendar features
- lag features
- rolling statistics
"""
import argparse
import pandas as pd
import os



def add_time_features(df, time_col):
    dt = pd.to_datetime(df[time_col])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    return df

def add_lags(df, col, lags=(1,2,3,6,12,24,48,72)):
    for l in lags:
        df[f"{col}_lag{l}"] = df[col].shift(l)
    return df

def add_rollings(df, col, windows=(3,6,12,24,48,72)):
    for w in windows:
        df[f"{col}_rollmean{w}"] = df[col].rolling(w).mean()
        df[f"{col}_rollstd{w}"] = df[col].rolling(w).std()
    return df

def main():
    
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--time_col", default="time")
    p.add_argument("--target_col", default="temperature_2m")
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    # Normalize expected column names
    if "datetime" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"datetime":"time"})
    if "relative_humidity_2m" in df.columns and "humidity" not in df.columns:
        df = df.rename(columns={"relative_humidity_2m":"humidity"})
    if "wind_speed_10m" in df.columns and "windspeed" not in df.columns:
        df = df.rename(columns={"wind_speed_10m":"windspeed"})

    df = add_time_features(df, args.time_col)
    df = add_lags(df, args.target_col)
    df = add_rollings(df, args.target_col)
    df = df.dropna().reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"Features saved to: {args.output} | rows: {len(df)}")

if __name__ == "__main__":
    main()