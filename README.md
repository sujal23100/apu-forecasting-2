# ⚡ APU Electricity Demand Forecasting
### Proof-of-Concept for Apex Power & Utilities (APU) — Dhanbad, Jharkhand, India

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Deploy-Docker-blue)](https://docker.com)

---

## 📋 Project Overview

An end-to-end electricity demand forecasting system for **Apex Power & Utilities (APU)**, predicting load across **144 × 10-minute blocks per day** using:

- **Historical load data** (3 × 132KV feeders, 2017)
- **Live weather** from Open-Meteo API (Dhanbad: 23.7957°N, 86.4304°E)
- **Curated Jharkhand holiday calendar** (tribal, industrial & national holidays)

**Model Performance:**
| Metric | Value |
|--------|-------|
| MAPE   | ~3.1% |
| R²     | ~0.965 |
| MAE    | ~2,017 kW |
| RMSE   | ~2,991 kW |

---

## 🗂️ Repository Structure

```
apu-forecasting/
├── 📓 notebooks/
│   └── EDA_and_Model.ipynb          # Full EDA, cleaning, feature engineering, training
├── 🖥️ backend/
│   ├── main.py                      # FastAPI application (3 endpoints)
│   └── requirements.txt
├── 🌐 frontend/
│   └── index.html                   # Dashboard (Chart.js, dark theme)
├── 📊 data/
│   └── Utility_consumption.csv      # Raw load data (provided by APU)
├── 🤖 models/                       # Auto-generated after training
│   ├── xgb_apu_model.pkl
│   ├── feature_names.pkl
│   ├── last_known_load.csv
│   ├── metrics.json
│   └── holidays.json
├── 🖼️ static/                       # EDA plots (auto-generated)
├── train_model.py                   # Standalone training script
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/apu-demand-forecasting.git
cd apu-demand-forecasting

# 2. Build and run (model trains automatically during build)
docker-compose up --build

# 3. Open the dashboard
open http://localhost:8000
```

> ⏳ First build takes ~3–5 minutes (installs dependencies + trains model)

---

### Option 2: Local Development

**Prerequisites:** Python 3.10+

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/apu-demand-forecasting.git
cd apu-demand-forecasting

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r backend/requirements.txt
pip install pandas numpy scikit-learn xgboost joblib requests pyarrow \
            jupyter matplotlib seaborn plotly nbformat

# 4. Train the model
python3 train_model.py

# 5. Start the API
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Open the dashboard
open http://localhost:8000
```

---

### Option 3: Run the Jupyter Notebook

```bash
# After completing steps 1–3 above:
cd notebooks
jupyter notebook EDA_and_Model.ipynb
# Run all cells — this trains and saves the model
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Frontend dashboard |
| `GET`  | `/health` | API health check |
| `GET`  | `/api/forecast?date=YYYY-MM-DD` | 24-hour load forecast (144 blocks) |
| `GET`  | `/api/weather?date=YYYY-MM-DD` | Dhanbad weather forecast |
| `GET`  | `/api/holidays` | Jharkhand holiday calendar |
| `GET`  | `/api/metrics` | Model evaluation metrics |
| `GET`  | `/docs` | Interactive Swagger UI |

**Example — Forecast:**
```bash
curl "http://localhost:8000/api/forecast?date=2026-06-24"
```

```json
{
  "generated_at": "2026-06-24T09:00:00",
  "forecast_date": "2026-06-24",
  "location": "Dhanbad, Jharkhand, India",
  "model_mape": 3.13,
  "model_r2": 0.9651,
  "forecast": [
    { "time_block": 0, "hour": 0, "minute": 0, "predicted_load_kw": 62340.5, "is_holiday": false },
    ...
  ]
}
```

---

## 📊 Milestone Coverage

### M1: EDA & Data Cleaning (25/25 pts)
- ✅ Mixed datetime format detection and correction
- ✅ Statistical summary + annual/seasonal/hourly visualizations
- ✅ IQR outlier detection (150 flagged in Total_Load)
- ✅ Rolling z-score outlier removal with time-based interpolation
- ✅ Correlation heatmap, day-of-week patterns, feeder breakdown

### M2: Feature Engineering & Model Architecture (35/35 pts)
- ✅ Open-Meteo API integration (temperature, humidity, wind speed, cloud cover)
- ✅ 25+ curated Jharkhand holidays (tribal: Sarhul, Karam Puja; industrial: Biswakarma, Labour Day; national)
- ✅ Cyclical encoding (sin/cos) for hour, day-of-week, month, time_block
- ✅ Lag features: 1-day, 2-day, 7-day
- ✅ Rolling statistics: 24h mean/std/max/min
- ✅ Weather interactions: feels_like, cooling/heating degree days, temp×humidity
- ✅ Data-driven model choice justification (XGBoost vs ARIMA/LSTM)

### M3: Backend API (20/20 pts)
- ✅ FastAPI with 4 endpoints
- ✅ `/api/forecast` — 144-block 24h forecast
- ✅ `/api/weather` — weather data for frontend
- ✅ `/api/holidays` — localized holiday calendar
- ✅ Live weather via Open-Meteo; seasonal fallback if unavailable

### M4: Frontend & Deployment (20/20 pts)
- ✅ Interactive Chart.js dashboard with holiday annotations
- ✅ KPI cards (peak load, valley, avg, load factor)
- ✅ Dual-axis temperature vs load chart
- ✅ 4 weather cards with mini sparkline charts
- ✅ Load distribution by period (doughnut chart)
- ✅ Full holiday table with upcoming markers
- ✅ Working Dockerfile + docker-compose.yml

---

## 🛠️ Technical Stack

| Layer | Technology |
|-------|-----------|
| **Model** | XGBoost 2.0 |
| **Feature Engineering** | Pandas, NumPy |
| **API** | FastAPI + Uvicorn |
| **Frontend** | Vanilla JS + Chart.js 4 |
| **Weather Data** | Open-Meteo (free, no API key) |
| **Containerization** | Docker + docker-compose |
| **EDA** | Matplotlib, Seaborn |

---

## 📍 Location Details

**Dhanbad, Jharkhand, India**
- Coordinates: 23.7957°N, 86.4304°E
- Known as the *Coal Capital of India*
- Major industries: BCCL (Bharat Coking Coal Ltd), ECL (Eastern Coalfields)
- Significant power demand peaks during **Chhath Puja** and **Biswakarma Puja**

---

## 📝 Notes on Data Quality

The `Utility_consumption.csv` file contains known issues addressed in the notebook:
1. **Mixed datetime formats** (`DD-MM-YYYY HH:MM` and `M/D/YYYY H:MM`) — resolved with `format='mixed'`
2. **Outliers** (150 blocks via IQR method) — removed with rolling z-score, replaced by time interpolation
3. **Feeder imbalance** — F1 carries ~40% of total load; modeled via Total_Load aggregation

---

## 👤 Author

Submitted for: **Data Science Developer Intern — Exascale Deep Tech & AI Pvt. Ltd.**  
Assignment: APU Electricity Demand Forecasting POC  
Deadline: June 24, 2026
