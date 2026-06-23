"""
APU Demand Forecasting API
FastAPI backend serving 24-hour load forecasts, weather, and holiday data
for Dhanbad, Jharkhand, India.
"""
import os, json, warnings
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────
app = FastAPI(
    title="APU Electricity Demand Forecasting API",
    description="Forecasts 10-minute electricity demand blocks for Apex Power & Utilities (APU), Dhanbad, Jharkhand.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
STATIC_DIR   = os.path.join(os.path.dirname(__file__), "..", "static")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─────────────────────────────────────────────
# Load Model Artifacts
# ─────────────────────────────────────────────
print("Loading model artifacts...")
model        = joblib.load(os.path.join(MODELS_DIR, "xgb_apu_model.pkl"))
FEATURES     = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
last_known   = pd.read_csv(os.path.join(MODELS_DIR, "last_known_load.csv"), parse_dates=["Datetime"])
with open(os.path.join(MODELS_DIR, "holidays.json")) as f:
    HOLIDAYS = json.load(f)
with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
    METRICS = json.load(f)
print("✅ Model loaded successfully")

DHANBAD_LAT, DHANBAD_LON = 23.7957, 86.4304

# ─────────────────────────────────────────────
# Helper: Fetch Forecast Weather from Open-Meteo
# ─────────────────────────────────────────────
def fetch_forecast_weather(days: int = 2) -> pd.DataFrame:
    """Fetch hourly forecast weather for Dhanbad, then resample to 10-min."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": DHANBAD_LAT, "longitude": DHANBAD_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
        "forecast_days": days,
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        wdf = pd.DataFrame({
            "Datetime":   pd.to_datetime(data["hourly"]["time"]),
            "w_temp":     data["hourly"]["temperature_2m"],
            "w_humidity": data["hourly"]["relative_humidity_2m"],
            "w_wind":     data["hourly"]["wind_speed_10m"],
            "w_cloud":    data["hourly"]["cloud_cover"],
        }).set_index("Datetime")
        return wdf.resample("10min").interpolate("linear")
    except Exception as e:
        print(f"Weather API error: {e} — using seasonal fallback")
        # Seasonal fallback for Dhanbad (June averages)
        now = datetime.now()
        idx = pd.date_range(now.replace(minute=0, second=0, microsecond=0), periods=days*144, freq="10min")
        hour_arr = idx.hour + idx.minute / 60
        temp  = 28 + 5  * np.sin(2 * np.pi * (hour_arr - 6) / 24)
        humid = 70 - 10 * np.sin(2 * np.pi * (hour_arr - 12) / 24)
        return pd.DataFrame({
            "w_temp": temp, "w_humidity": humid,
            "w_wind": 12.0, "w_cloud": 40.0
        }, index=idx)


# ─────────────────────────────────────────────
# Helper: Build Feature Matrix for Forecast
# ─────────────────────────────────────────────
def build_forecast_features(target_times: pd.DatetimeIndex, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Construct the full feature matrix for inference."""
    df = pd.DataFrame({"Datetime": target_times})
    df = df.set_index("Datetime")

    # Merge weather
    df = df.join(weather_df[["w_temp","w_humidity","w_wind","w_cloud"]], how="left")
    df = df.ffill().bfill()
    df = df.fillna({"w_temp": 28, "w_humidity": 70, "w_wind": 12, "w_cloud": 40})

    df = df.reset_index()
    df.rename(columns={"index": "Datetime"}, inplace=True)

    # Time features
    df["hour"]        = df["Datetime"].dt.hour
    df["minute"]      = df["Datetime"].dt.minute
    df["time_block"]  = df["hour"] * 6 + df["minute"] // 10
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["day_of_year"] = df["Datetime"].dt.dayofyear
    df["month"]       = df["Datetime"].dt.month
    df["week"]        = df["Datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = df["month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

    # Cyclical encoding
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]   = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["day_of_week"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["block_sin"] = np.sin(2*np.pi*df["time_block"]/144)
    df["block_cos"] = np.cos(2*np.pi*df["time_block"]/144)

    # Holidays
    holiday_dates = set(pd.to_datetime(list(HOLIDAYS.keys())).normalize())
    df["is_holiday"] = df["Datetime"].dt.normalize().isin(holiday_dates).astype(int)

    # Weather interactions
    df["temp_humidity"]  = df["w_temp"] * df["w_humidity"]
    df["feels_like"]     = df["w_temp"] - 0.4*(df["w_temp"]-10)*(1-df["w_humidity"]/100)
    df["cooling_degree"] = np.maximum(df["w_temp"] - 24, 0)
    df["heating_degree"] = np.maximum(18 - df["w_temp"], 0)

    # Lag features from last_known history
    history = last_known.copy()
    history["Datetime"] = pd.to_datetime(history["Datetime"])
    history = history.sort_values("Datetime").reset_index(drop=True)

    def get_lag(dt, days):
        target = dt - pd.Timedelta(days=days)
        match = history[history["Datetime"] == target]
        if not match.empty:
            return match["Total_Load"].values[0]
        # fallback: same time_block average from history
        block = dt.hour * 6 + dt.minute // 10
        same_block = history[
            (history["Datetime"].dt.hour * 6 + history["Datetime"].dt.minute // 10) == block
        ]["Total_Load"]
        return same_block.mean() if not same_block.empty else history["Total_Load"].mean()

    df["lag_1d"] = df["Datetime"].apply(lambda x: get_lag(x, 1))
    df["lag_2d"] = df["Datetime"].apply(lambda x: get_lag(x, 2))
    df["lag_7d"] = df["Datetime"].apply(lambda x: get_lag(x, 7))

    hist_load = history["Total_Load"].values[-144:]
    df["roll_mean_24h"] = np.mean(hist_load)
    df["roll_std_24h"]  = np.std(hist_load)
    df["roll_max_24h"]  = np.max(hist_load)
    df["roll_min_24h"]  = np.min(hist_load)

    return df


# ─────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────
class ForecastBlock(BaseModel):
    datetime: str
    time_block: int
    hour: int
    minute: int
    predicted_load_kw: float
    is_holiday: bool
    holiday_name: Optional[str] = None

class WeatherPoint(BaseModel):
    datetime: str
    temperature: float
    humidity: float
    wind_speed: float
    cloud_cover: float

class HolidayEntry(BaseModel):
    date: str
    name: str
    is_upcoming: bool

class ForecastResponse(BaseModel):
    generated_at: str
    forecast_date: str
    location: str
    model_mape: float
    model_r2: float
    forecast: List[ForecastBlock]

class WeatherResponse(BaseModel):
    location: str
    forecast_date: str
    weather: List[WeatherPoint]

class HolidayResponse(BaseModel):
    location: str
    total_holidays: int
    upcoming_in_30_days: List[HolidayEntry]
    all_holidays: List[HolidayEntry]


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgb_apu_model", "version": "1.0.0"}

@app.get("/api/metrics")
def get_metrics():
    """Return model evaluation metrics."""
    return METRICS


@app.get("/api/forecast", response_model=ForecastResponse)
def get_forecast(date: Optional[str] = None):
    """
    Generate a 24-hour load forecast (144 × 10-minute blocks).
    
    - **date**: ISO date string (YYYY-MM-DD). Defaults to today.
    """
    if date:
        try:
            forecast_date = pd.Timestamp(date)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    else:
        forecast_date = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)

    target_times = pd.date_range(forecast_date, periods=144, freq="10min")

    # Fetch weather
    weather_df = fetch_forecast_weather(days=2)

    # Build features & predict
    feat_df = build_forecast_features(target_times, weather_df)
    X = feat_df[FEATURES]
    preds = model.predict(X)
    preds = np.maximum(preds, 0)  # no negative load

    # Holiday map
    h_map = {pd.Timestamp(k).normalize(): v for k, v in HOLIDAYS.items()}

    blocks = []
    for i, (ts, pred) in enumerate(zip(target_times, preds)):
        hol = h_map.get(ts.normalize(), None)
        blocks.append(ForecastBlock(
            datetime=ts.isoformat(),
            time_block=i,
            hour=ts.hour,
            minute=ts.minute,
            predicted_load_kw=round(float(pred), 2),
            is_holiday=hol is not None,
            holiday_name=hol
        ))

    return ForecastResponse(
        generated_at=datetime.now().isoformat(),
        forecast_date=forecast_date.date().isoformat(),
        location="Dhanbad, Jharkhand, India",
        model_mape=METRICS["MAPE"],
        model_r2=METRICS["R2"],
        forecast=blocks
    )


@app.get("/api/weather", response_model=WeatherResponse)
def get_weather(date: Optional[str] = None):
    """
    Return 24-hour weather forecast for Dhanbad, Jharkhand.
    
    - **date**: ISO date string. Defaults to today.
    """
    if date:
        forecast_date = pd.Timestamp(date)
    else:
        forecast_date = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)

    weather_df = fetch_forecast_weather(days=2).reset_index()
    weather_df.rename(columns={"index": "Datetime"}, inplace=True)

    day_weather = weather_df[
        weather_df["Datetime"].dt.normalize() == forecast_date
    ].head(144)

    if day_weather.empty:
        day_weather = weather_df.head(144)

    points = []
    for _, row in day_weather.iterrows():
        points.append(WeatherPoint(
            datetime=row["Datetime"].isoformat(),
            temperature=round(float(row["w_temp"]), 1),
            humidity=round(float(row["w_humidity"]), 1),
            wind_speed=round(float(row["w_wind"]), 1),
            cloud_cover=round(float(row["w_cloud"]), 1),
        ))

    return WeatherResponse(
        location="Dhanbad, Jharkhand, India (23.7957°N, 86.4304°E)",
        forecast_date=forecast_date.date().isoformat(),
        weather=points
    )


@app.get("/api/holidays", response_model=HolidayResponse)
def get_holidays():
    """
    Return the curated Dhanbad/Jharkhand holiday calendar.
    Includes tribal, industrial, and national holidays.
    """
    today = datetime.now().date()
    upcoming = []
    all_h = []

    for date_str, name in sorted(HOLIDAYS.items()):
        hdate = datetime.strptime(date_str, "%Y-%m-%d").date()
        is_upcoming = 0 <= (hdate - today).days <= 30
        entry = HolidayEntry(date=date_str, name=name, is_upcoming=is_upcoming)
        all_h.append(entry)
        if is_upcoming:
            upcoming.append(entry)

    return HolidayResponse(
        location="Dhanbad, Jharkhand, India",
        total_holidays=len(all_h),
        upcoming_in_30_days=upcoming,
        all_holidays=all_h
    )
