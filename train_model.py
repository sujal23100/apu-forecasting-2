"""
APU Demand Forecasting - Model Training Script
Trains XGBoost model on historical data and saves artifact
"""
import pandas as pd
import numpy as np
import requests
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('/home/claude/apu-forecasting/data/Utility_consumption.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', dayfirst=True)
df = df.sort_values('Datetime').reset_index(drop=True)

# Full datetime index (no gaps)
full_range = pd.date_range(start=df['Datetime'].min(), end=df['Datetime'].max(), freq='10min')
df = df.set_index('Datetime').reindex(full_range)
df.index.name = 'Datetime'

# Total load
df['Total_Load'] = df['F1_132KV_PowerConsumption'] + df['F2_132KV_PowerConsumption'] + df['F3_132KV_PowerConsumption']

# Outlier detection & cleaning using rolling IQR
def clean_outliers(series, window=144):
    """Replace outliers (3-sigma rolling) with interpolated values"""
    s = series.copy()
    roll_mean = s.rolling(window, center=True, min_periods=1).mean()
    roll_std  = s.rolling(window, center=True, min_periods=1).std()
    lower = roll_mean - 3 * roll_std
    upper = roll_mean + 3 * roll_std
    mask = (s < lower) | (s > upper)
    s[mask] = np.nan
    return s

for col in ['F1_132KV_PowerConsumption','F2_132KV_PowerConsumption','F3_132KV_PowerConsumption','Total_Load']:
    df[col] = clean_outliers(df[col])

# Interpolate remaining NaNs
df = df.interpolate(method='time').ffill().bfill()
print(f"Data cleaned: {len(df)} rows, NaNs remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 2. FETCH HISTORICAL WEATHER (Open-Meteo)
# ─────────────────────────────────────────────
print("Fetching historical weather data from Open-Meteo...")
DHANBAD_LAT, DHANBAD_LON = 23.7957, 86.4304

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": DHANBAD_LAT, "longitude": DHANBAD_LON,
    "start_date": "2017-01-01", "end_date": "2017-12-30",
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
    "timezone": "Asia/Kolkata"
}
try:
    resp = requests.get(url, params=params, timeout=30)
    weather_json = resp.json()
    weather_hourly = pd.DataFrame({
        'Datetime': pd.to_datetime(weather_json['hourly']['time']),
        'w_temp':    weather_json['hourly']['temperature_2m'],
        'w_humidity':weather_json['hourly']['relative_humidity_2m'],
        'w_wind':    weather_json['hourly']['wind_speed_10m'],
        'w_cloud':   weather_json['hourly']['cloud_cover'],
    })
    weather_hourly = weather_hourly.set_index('Datetime')
    # Resample to 10-min
    weather_10min = weather_hourly.resample('10min').interpolate('linear')
    df = df.join(weather_10min, how='left')
    df[['w_temp','w_humidity','w_wind','w_cloud']] = df[['w_temp','w_humidity','w_wind','w_cloud']].interpolate('linear').ffill().bfill()
    print("Weather data integrated successfully!")
except Exception as e:
    print(f"Weather fetch failed ({e}), using CSV weather columns as fallback")
    df['w_temp']    = df['Temperature']
    df['w_humidity']= df['Humidity']
    df['w_wind']    = df['WindSpeed']
    df['w_cloud']   = 50.0  # fallback constant

# ─────────────────────────────────────────────
# 3. DHANBAD HOLIDAYS (Jharkhand-specific)
# ─────────────────────────────────────────────
print("Building Dhanbad/Jharkhand holiday calendar...")

# National + Jharkhand state + Industrial/Mining holidays
jharkhand_holidays_2017 = {
    # National
    '2017-01-26': 'Republic Day',
    '2017-08-15': 'Independence Day',
    '2017-10-02': 'Gandhi Jayanti',
    # Jharkhand Foundation Day
    '2017-11-15': 'Jharkhand Foundation Day',
    # Hindu Festivals
    '2017-01-14': 'Makar Sankranti / Tusu Puja',
    '2017-03-13': 'Holi',
    '2017-03-29': 'Ram Navami',
    '2017-04-14': 'Ambedkar Jayanti / Baisakhi',
    '2017-06-26': 'Rath Yatra',
    '2017-08-07': 'Sarhul (Jharkhand Tribal)',
    '2017-08-25': 'Karam Puja (Jharkhand Tribal)',
    '2017-09-05': 'Teej',
    '2017-09-28': 'Navratri Begins',
    '2017-10-07': 'Durga Puja Ashtami',
    '2017-10-08': 'Dussehra',
    '2017-10-19': 'Diwali',
    '2017-10-20': 'Govardhan Puja',
    '2017-11-03': 'Chhath Puja (Major in Jharkhand)',
    '2017-11-04': 'Chhath Puja Day 2',
    '2017-11-13': 'Dev Diwali',
    '2017-12-16': 'Santhali New Year / Soharai',
    '2017-12-25': 'Christmas',
    # Muslim Festivals (approx dates 2017)
    '2017-06-26': 'Eid ul-Fitr',
    '2017-09-02': 'Eid ul-Adha',
    # Industrial/Mining (BCCL, ECL shutdowns common in Dhanbad)
    '2017-01-01': 'New Year',
    '2017-05-01': 'Labour Day (Mine Workers)',
    '2017-09-17': 'Biswakarma Puja (Industrial shutdown)',
    '2017-10-19': 'Deepawali Industrial Shutdown',
}
holiday_dates = pd.to_datetime(list(jharkhand_holidays_2017.keys()))

df['is_holiday'] = df.index.normalize().isin(holiday_dates).astype(int)
df['holiday_name'] = df.index.normalize().map(
    {pd.Timestamp(k): v for k, v in jharkhand_holidays_2017.items()}
).fillna('')

print(f"  → {len(jharkhand_holidays_2017)} holidays loaded, {df['is_holiday'].sum()} 10-min blocks flagged")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("Engineering features...")

df = df.reset_index()
df.rename(columns={'index': 'Datetime'}, inplace=True)

# Time features
df['hour']        = df['Datetime'].dt.hour
df['minute']      = df['Datetime'].dt.minute
df['time_block']  = df['hour'] * 6 + df['minute'] // 10  # 0-143
df['day_of_week'] = df['Datetime'].dt.dayofweek           # 0=Mon
df['day_of_year'] = df['Datetime'].dt.dayofyear
df['month']       = df['Datetime'].dt.month
df['week']        = df['Datetime'].dt.isocalendar().week.astype(int)
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
df['season']      = df['month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

# Cyclical encoding (prevent discontinuity at boundaries)
df['hour_sin']      = np.sin(2*np.pi*df['hour']/24)
df['hour_cos']      = np.cos(2*np.pi*df['hour']/24)
df['dow_sin']       = np.sin(2*np.pi*df['day_of_week']/7)
df['dow_cos']       = np.cos(2*np.pi*df['day_of_week']/7)
df['month_sin']     = np.sin(2*np.pi*df['month']/12)
df['month_cos']     = np.cos(2*np.pi*df['month']/12)
df['block_sin']     = np.sin(2*np.pi*df['time_block']/144)
df['block_cos']     = np.cos(2*np.pi*df['time_block']/144)

# Weather interaction features
df['temp_humidity']  = df['w_temp'] * df['w_humidity']
df['feels_like']     = df['w_temp'] - 0.4 * (df['w_temp'] - 10) * (1 - df['w_humidity']/100)
df['cooling_degree'] = np.maximum(df['w_temp'] - 24, 0)  # above comfort zone
df['heating_degree'] = np.maximum(18 - df['w_temp'], 0)  # below comfort zone

# Lag features (previous day same block, previous week same block)
df = df.sort_values('Datetime').reset_index(drop=True)
df['lag_1d']  = df['Total_Load'].shift(144)   # 1 day = 144 blocks
df['lag_7d']  = df['Total_Load'].shift(1008)  # 7 days
df['lag_2d']  = df['Total_Load'].shift(288)

# Rolling statistics
df['roll_mean_24h'] = df['Total_Load'].shift(1).rolling(144, min_periods=1).mean()
df['roll_std_24h']  = df['Total_Load'].shift(1).rolling(144, min_periods=1).std().fillna(0)
df['roll_max_24h']  = df['Total_Load'].shift(1).rolling(144, min_periods=1).max()
df['roll_min_24h']  = df['Total_Load'].shift(1).rolling(144, min_periods=1).min()

# Drop initial NaN rows (due to lags)
df = df.dropna(subset=['lag_7d']).reset_index(drop=True)
print(f"Feature engineering done. Dataset: {df.shape}")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT (last 30 days = test)
# ─────────────────────────────────────────────
split_date = df['Datetime'].max() - pd.Timedelta(days=30)
train = df[df['Datetime'] <= split_date]
test  = df[df['Datetime'] >  split_date]
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

FEATURES = [
    'hour','minute','time_block','day_of_week','day_of_year','month','week',
    'is_weekend','is_holiday','season',
    'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','block_sin','block_cos',
    'w_temp','w_humidity','w_wind','w_cloud',
    'temp_humidity','feels_like','cooling_degree','heating_degree',
    'lag_1d','lag_2d','lag_7d',
    'roll_mean_24h','roll_std_24h','roll_max_24h','roll_min_24h'
]
TARGET = 'Total_Load'

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# ─────────────────────────────────────────────
# 6. TRAIN XGBOOST MODEL
# ─────────────────────────────────────────────
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric='rmse'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# ─────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2   = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  MAE  : {mae:,.2f} kW")
print(f"  RMSE : {rmse:,.2f} kW")
print(f"  MAPE : {mape:.2f}%")
print(f"  R²   : {r2:.4f}")
print(f"{'='*50}")

# ─────────────────────────────────────────────
# 8. SAVE ARTIFACTS
# ─────────────────────────────────────────────
joblib.dump(model,    '/home/claude/apu-forecasting/models/xgb_apu_model.pkl')
joblib.dump(FEATURES, '/home/claude/apu-forecasting/models/feature_names.pkl')

# Save a "last known values" snapshot for inference-time lag features
last_known = df.tail(1008)[['Datetime','Total_Load']].copy()
last_known.to_parquet('/home/claude/apu-forecasting/models/last_known_load.parquet', index=False)

# Save metrics
metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
with open('/home/claude/apu-forecasting/models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nModel artifacts saved!")
print("  → models/xgb_apu_model.pkl")
print("  → models/feature_names.pkl")
print("  → models/last_known_load.parquet")
print("  → models/metrics.json")
