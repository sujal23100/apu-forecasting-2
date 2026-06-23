# ─────────────────────────────────────────────────────────────────────────────
# APU Electricity Demand Forecasting — Dockerfile
# Single-container deployment: trains model + serves FastAPI + React frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       pandas numpy scikit-learn xgboost joblib requests pyarrow \
       jupyter nbformat matplotlib seaborn plotly

# Copy entire project
COPY . .

# Run model training at build time (so the container is ready to serve)
RUN python3 train_model.py

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]