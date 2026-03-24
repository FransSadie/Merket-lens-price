# Market Lens (Price-Only Model)

Market Lens is a price-focused market prediction project.

The active system does this:
- ingests market prices from Yahoo Finance
- builds price-only feature snapshots
- generates future-return labels
- trains a baseline prediction model
- stores model history so every retrain is tracked digitally
- serves a dashboard for running and monitoring the workflow

News-related code remains in the repo as archived experimental work, but it is not part of the active default pipeline.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Recommended `.env` settings for the active price-only workflow:
- `APP_NAME=Market Lens`
- `MARKET_TICKERS=AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,SPY,QQQ,IWM,DIA`
- `MARKET_HISTORY_PERIOD=1y`
- `MARKET_HISTORY_INTERVAL=1d`
- `PREDICT_HOLD_THRESHOLD=0.6`
- `ENABLE_NEWS_PIPELINE=false`
- `TRAINING_FEATURE_SET=price_only`
- `TRAINING_HORIZON_DAYS=3`
- `TRAINING_TARGET_RETURN_THRESHOLD=0.002`

## 2) Run API

```bash
uvicorn app.main:app --reload
```

Shortcut:

```bash
npm start
```

PowerShell helper:

```powershell
.\scripts\start_api.ps1
```

## 3) Run frontend dashboard

```bash
cd frontend
npm install
npm run dev
```

Shortcut from project root:

```bash
npm run frontend
```

Open:
- `http://127.0.0.1:5173`

The dashboard gives you:
- market ingestion controls
- feature/label pipeline controls
- model training and prediction
- model run history and metric comparison
- data status and data quality views
- prediction logs
- embedded docs

## 4) Run market ingestion

```bash
python -m app.ingestion.run_once
```

Or via API:
- `POST /ingest/run`

## 5) Run active feature + label pipeline

```bash
python -m app.features.run_once
```

PowerShell helper:

```powershell
.\scripts\run_feature_pipeline.ps1
```

API endpoints:
- `POST /pipeline/run`
- `GET /pipeline/status`
- `GET /data/status`
- `GET /data/quality`

## 6) Train model

```bash
python -m app.models.train_baseline
```

PowerShell helper:

```powershell
.\scripts\run_model_training.ps1
```

Artifacts are saved in `./artifacts`:
- `baseline_model.joblib`
- `baseline_metadata.json`
- `baseline_model_<version>.joblib`
- `baseline_metadata_<version>.json`
- `current_model.json`

API endpoints:
- `POST /model/train`
- `GET /model/status`
- `GET /model/history?limit=20`
- `GET /predict?ticker=SPY`
- `GET /prediction/logs?limit=100`
- `POST /run/full?train_model=true`

## 7) Daily runbook

PowerShell helpers:
- full refresh + train:
  - `.\scripts\run_full_refresh.ps1`
- data status/quality report:
  - `.\scripts\show_data_checks.ps1`
- local one-off prediction:
  - `.\scripts\predict_ticker.ps1 -Ticker SPY`

## 8) Active data model

Main active tables:
- `market_prices`
  - OHLCV rows per ticker and day
- `feature_snapshots`
  - model input rows built from price action and retained compatibility fields
- `market_labels`
  - future direction / return targets
- `model_runs`
  - stored retrain history for comparing models over time
- `prediction_logs`
  - stored prediction requests/results
- `ingestion_runs`
  - audit log for pipeline jobs

Key active price-only features include:
- `return_1d`
- `price_return_3d`
- `price_return_5d`
- `price_return_10d`
- `price_return_20d`
- `ma_gap_5d`
- `ma_gap_20d`
- `ma_crossover_5_20`
- `range_pct_1d`
- `atr_14_pct`
- `rolling_volatility_20d`
- `volatility_regime_60d`
- `volume_zscore_20d`
- `volume_change_5d`

## 9) Current operating mode

This repo is intentionally price-only focused:
- news ingestion is disabled by default
- NLP is not part of the active workflow
- training defaults to `price_only`
- model history is stored automatically after each retrain

## 10) Archived news work

The repo still contains older news-related modules and historical import tools.
They are retained for reference and future experimentation, but they are not part of the active default workflow.

## 11) Next step

Best next steps for this repo:
1. tune price-only features further
2. compare multiple horizon/threshold combinations
3. add lightweight backtesting and paper-trade style evaluation
4. only revisit news if it starts beating the price-only benchmark again
