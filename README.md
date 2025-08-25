# Weather Forecasting with Python

A practical project for **time series forecasting** of weather (temperature by default) using feature engineering and machine learning models.  

It includes:
- **RandomForest** baseline (quick & simple).
- **XGBoost** for stronger accuracy (with Early Stopping support).
- (Optional) script for randomized hyperparameter search.

The project is modular and easily extensible to forecast other variables (humidity, wind speed, precipitation, …).

---

## 📂 Project Structure
```
weather-forecasting/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  │  └─ sample_city_hourly.csv     # synthetic demo dataset (~60 days hourly)
│  ├─ processed/
├─ models/
├─ src/
│  ├─ fetch_data.py        # fetch real data from Open-Meteo
│  ├─ make_features.py     # create lags/rolling features + calendar features
│  ├─ train.py             # RandomForest baseline
│  ├─ train_xgb.py         # XGBoost model (stronger)
│  ├─ tune_xgb.py          # optional hyperparameter tuning
│  ├─ predict.py           # generate future forecasts
│  ├─ evaluate.py          # display saved metrics
│  └─ utils.py
```

---

## 🚀 Quickstart (using included synthetic data)

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Generate features:
```bash
python src/make_features.py --input data/raw/sample_city_hourly.csv --output data/processed/sample_features.csv --target_col temperature_2m
```

3. Train a stronger model (XGBoost):
```bash
python src/train_xgb.py --data data/processed/sample_features.csv --target_col temperature_2m --save_path models/xgb_temp.pkl
```

4. (Optional) Compare with RandomForest:
```bash
python src/train.py --data data/processed/sample_features.csv --target_col temperature_2m --save_path models/rf_temp.pkl
```

5. Check evaluation metrics:
```bash
python src/evaluate.py --metrics_path models/metrics_xgb.json
python src/evaluate.py --metrics_path models/metrics.json
```

6. Forecast the next 24 hours:
```bash
python src/predict.py --recent_csv data/raw/sample_city_hourly.csv --model_path models/xgb_temp.pkl --horizon 24
```

---

## 🌍 Fetch Real Weather Data (Open-Meteo)

You can fetch real historical hourly weather data for any location (no API key required).  
Example: Riyadh (KSA)
```bash
python src/fetch_data.py --lat 24.7136 --lon 46.6753 --start 2024-01-01 --end 2024-12-31 --outfile data/raw/riyadh_hourly.csv

python src/make_features.py --input data/raw/riyadh_hourly.csv --output data/processed/riyadh_features.csv --target_col temperature_2m

python src/train_xgb.py --data data/processed/riyadh_features.csv --save_path models/xgb_riyadh.pkl

python src/predict.py --recent_csv data/raw/riyadh_hourly.csv --model_path models/xgb_riyadh.pkl --horizon 24
```

---

## 🔧 Hyperparameter Tuning (Optional)
Run randomized search over XGBoost parameters:
```bash
python src/tune_xgb.py --data data/processed/sample_features.csv --target_col temperature_2m --n_iter 40
```
Results and best parameters will be saved to:
- `models/xgb_tuned.pkl`
- `models/xgb_tuned_report.json`

---

## 📊 Ideas for Improvement
- Add more lags/rolling windows (e.g., up to 168 hours = 1 week).
- Add richer calendar features (weekends, holidays, seasons).
- Train for multiple targets (humidity, wind, precipitation).
- Experiment with deep learning models (LSTM, Transformers).
- Use Optuna or GridSearch for smarter hyperparameter tuning.

---

## 📜 License
This project is MIT licensed. Feel free to use, modify, and share.  

---

## ⭐ Contribute
If you like this project, consider giving it a ⭐ on GitHub!