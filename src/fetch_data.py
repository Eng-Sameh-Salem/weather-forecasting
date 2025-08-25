# -*- coding: utf-8 -*-
"""
Fetch historical hourly weather data from Open-Meteo.
Docs: https://open-meteo.com/
No API key required.
"""
import argparse
import requests
import pandas as pd

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_hourly(lat, lon, start, end, variables=None, timezone="auto"):
    if variables is None:
        variables = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": timezone
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    hourly = data.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError("No 'hourly' key in API response.")
    df = pd.DataFrame(hourly)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--outfile", type=str, required=True)
    args = p.parse_args()

    df = fetch_hourly(args.lat, args.lon, args.start, args.end)
    df.rename(columns={"relative_humidity_2m":"humidity","wind_speed_10m":"windspeed"}, inplace=True)
    df.to_csv(args.outfile, index=False)
    print(f"Saved data to: {args.outfile} | rows: {len(df)}")

if __name__ == "__main__":
    main()