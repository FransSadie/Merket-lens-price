# Market Lens (Price-Only Model)

Market Lens is a price-focused market prediction project.

The active system does this:
- ingests market prices from Yahoo Finance
- builds richer price-only feature snapshots
- generates future-return labels
- trains an active LightGBM prediction model
- runs an internal logistic-regression benchmark on every retrain
- records candidate LightGBM trials, threshold analysis, and feature importance
- stores model history so every retrain is tracked digitally
- serves a dashboard for running and monitoring the workflow

News-related code remains in the repo as archived experimental work, but it is not part of the active default pipeline.
Transformer work is deferred until the dataset is redesigned around sequential windows rather than one aggregated snapshot row per ticker/date.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Recommended `.env` settings for the active price-only workflow:
- `APP_NAME=Market Lens`
- `MARKET_TICKERS=AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD,AVGO,CRM,ORCL,ADBE,NFLX,CSCO,QCOM,JPM,GS,MS,V,MA,UNH,LLY,COST,WMT,KO,PEP,XOM,CVX,CAT,BA,GE,SPY,QQQ,IWM,DIA,XLF,XLK,XLV,XLY,XLI,XLE,XLB,XLU,XLP,TLT,GLD`
- `MARKET_HISTORY_PERIOD=5y`
- `MARKET_HISTORY_INTERVAL=1d`
- `MODEL_TYPE=lightgbm`
- `PREDICT_HOLD_THRESHOLD=0.6`
- `ENABLE_NEWS_PIPELINE=false`
- `TRAINING_FEATURE_SET=price_only`
- `TRAINING_HORIZON_DAYS=5`
- `TRAINING_TARGET_RETURN_THRESHOLD=0.004`

LightGBM defaults exposed in config:
- `LIGHTGBM_NUM_LEAVES=15`
- `LIGHTGBM_LEARNING_RATE=0.05`
- `LIGHTGBM_N_ESTIMATORS=200`
- `LIGHTGBM_MAX_DEPTH=6`
- `LIGHTGBM_MIN_CHILD_SAMPLES=20`
- `LIGHTGBM_SUBSAMPLE=0.9`
- `LIGHTGBM_COLSAMPLE_BYTREE=0.9`

## 2) Run API

```bash
uvicorn app.main:app --reload
```

Shortcut:

```bash
npm start
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
- LightGBM vs logistic benchmark visibility
- selected LightGBM trial visibility
- threshold sweep visibility
- experiment matrix visibility across 1d/3d/5d horizons and 0.2%/0.4%/0.6% targets
- lightweight strategy scorecard visibility
- top feature-driver visibility
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

This now computes a wider pure-price feature set including:
- multi-horizon returns: `2d`, `3d`, `5d`, `10d`, `15d`, `20d`
- moving-average and EMA gaps/crossovers
- ATR / range features
- standard and downside volatility
- RSI, Bollinger z-score, breakout, drawdown
- trend slopes and up-day ratio
- richer volume regime features

## 6) Train model

```bash
python -m app.models.train_baseline
```

Artifacts are saved in `./artifacts`:
- `baseline_model.joblib`
- `baseline_metadata.json`
- `baseline_model_<version>.joblib`
- `baseline_metadata_<version>.json`
- `current_model.json`

The active artifact is LightGBM by default.
Logistic regression is still trained inside the routine as a benchmark and is recorded in metadata/history, but it is no longer the active saved model path.

The trainer now also records:
- LightGBM candidate trial results
- chosen trial parameters
- threshold sweep analysis
- top feature importance rows
- constrained strategy diagnostics with capped daily positions and per-ticker cooldowns

Experiment matrix endpoints:
- `POST /model/experiments/run`
- `GET /model/experiments/status`

API endpoints:
- `POST /model/train`
- `GET /model/status`
- `GET /model/history?limit=20`
- `GET /predict?ticker=SPY`
- `GET /prediction/logs?limit=100`
- `POST /run/full?train_model=true`

## 7) Current operating mode

This repo is intentionally price-only focused:
- news ingestion is disabled by default
- NLP is not part of the active workflow
- training defaults to `price_only`
- LightGBM is the active model type
- logistic regression remains an internal benchmark
- model history is stored automatically after each retrain

## 8) Transformer note

Transformer work is intentionally deferred.
If we revisit that path later, it should be built on a new sequence-style dataset builder rather than the current one-row-per-snapshot setup.

## 9) Current reality

The current richer LightGBM pipeline is operational end to end on a broader 5-year, 46-ticker dataset with relative-strength features against market and sector benchmarks. The default active target is now aligned to the best recent matrix result: 5-day horizon, 0.4% move threshold. In the latest run, LightGBM now slightly beats the logistic benchmark on validation accuracy, F1, and ROC AUC. The repo is in a much healthier place for real model comparison than the earlier small-sample runs.

## 10) Next step

Best next steps for this repo:
1. use the dashboard experiment matrix to compare horizon / threshold combinations on the richer feature set
2. keep retraining and use the dashboard to compare LightGBM against logistic fairly
3. tighten strategy realism further if we want broker-style backtesting later
4. only revisit transformer work after sequence data is introduced

