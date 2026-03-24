# Market Lens (Price-First Baseline Model)

This project starts with a data-ingestion foundation for:
- Market price ingestion from Yahoo Finance
- Price-first feature snapshots + labels for baseline modeling
- Baseline model training + prediction endpoint
- News tooling retained in the repo but disabled from the active pipeline by default
- Data quality checks + walk-forward/baseline evaluation metadata
- React + Vite + Tailwind operator dashboard

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Recommended coverage settings in `.env` for better model training:
- `MARKET_TICKERS=AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,SPY,QQQ,IWM,DIA`
- `MARKET_HISTORY_PERIOD=1y`
- `MARKET_HISTORY_INTERVAL=1d`
- `NEWS_PAGE_SIZE=100`
- `NEWS_MAX_PAGES=3`
- `NEWS_LOOKBACK_DAYS=7`
- `HISTORICAL_NEWS_DIR=./data/historical_news`
- `HISTORICAL_NEWS_FORMAT=csv`
- `HISTORICAL_NEWS_BATCH_LIMIT=0`
- `GDELT_GKG_RAW_DIR=./data/gdelt_gkg_raw`
- `GDELT_GKG_NORMALIZED_DIR=./data/historical_news`
- `GDELT_GKG_BATCH_LIMIT=0`
- `GDELT_GKG_ROW_LIMIT=0`
- `PREDICT_HOLD_THRESHOLD=0.6`
- `ENABLE_NEWS_PIPELINE=false`
- `TRAINING_FEATURE_SET=price_only`
- `TRAINING_HORIZON_DAYS=3`
- `TRAINING_TARGET_RETURN_THRESHOLD=0.002`

## 2) Run API + scheduler

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

On startup, tables are created and the scheduler starts.

## 3) Run frontend dashboard (React + Vite + Tailwind)

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

The dashboard calls the API through Vite proxy and exposes:
- pipeline control buttons
- model training/predict
- model history + metric comparison after each retrain
- data quality/status views
- prediction log table
- embedded `README.md` and `PROJECT_OVERVIEW.txt` text
- tooltips, confirm prompts, auto-refresh controls, KPI badges, filtered logs, and copy-to-clipboard JSON blocks

## 4) Trigger ingestion manually

```bash
python -m app.ingestion.run_once
```

Or test API endpoints (run in a second terminal while API is running):

```powershell
.\scripts\smoke_api.ps1
```

Historical batch import:

```bash
python -m app.ingestion.run_historical_import
```

PowerShell helper:

```powershell
.\scripts\run_historical_import.ps1
```

Expected historical file schema for `csv`:
- required: `title`, `url`
- recommended: `published_at`, `source_name`, `description`, `content`, `external_id`
- accepted aliases include:
  - `headline` -> `title`
  - `link` or `DocumentIdentifier` -> `url`
  - `date` or `datetime` -> `published_at`
  - `source` or `SourceCommonName` -> `source_name`

Drop batch files into `data/historical_news/`.

GDELT GKG normalization:

```bash
python -m app.ingestion.run_gdelt_normalize
```

PowerShell helper:

```powershell
.\scripts\run_gdelt_normalize.ps1
```

Workflow for raw GDELT GKG files:
- drop raw `.zip`, `.csv`, `.txt`, or `.tsv` files into `data/gdelt_gkg_raw/`
- run the normalizer
- it writes importer-ready `.normalized.csv` files into `data/historical_news/`
- then run the historical importer
- rerunning the historical importer now refreshes existing `historical_batch` rows instead of only skipping duplicates

Normalized GKG mapping:
- `DocumentIdentifier` -> `url`
- `SourceCommonName` -> `source_name`
- `DATE` -> `published_at`
- `GKGRECORDID` -> `external_id`
- `title`, `description`, and `content` are synthesized from cleaned themes, organizations, persons, counts, and tone so the existing NLP pipeline has more usable text to process

## 5) Health check

`GET http://127.0.0.1:8000/health`


## 6) Run active feature + label pipeline

```bash
python -m app.features.run_once
```

PowerShell helper:

```powershell
.\scripts\run_feature_pipeline.ps1
```

API option (while server is running):

- `POST /pipeline/run`
- `POST /maintenance/seed-article-hashes`
- `POST /maintenance/seed-nlp-markers`
- `GET /pipeline/status`
- `GET /data/status`
- `GET /data/quality`
- `POST /ingest/historical/run`
- `POST /normalize/gdelt/run`

For larger historical backfills, run:

```bash
python -m app.ingestion.run_once
python -m app.features.run_once
```

## 7) Train baseline model

```bash
python -m app.models.train_baseline
```

PowerShell helper:

```powershell
.\scripts\run_model_training.ps1
```

Artifacts are saved in `./artifacts` by default:
- `baseline_model.joblib` (latest pointer)
- `baseline_metadata.json` (latest pointer)
- `baseline_model_<version>.joblib` (versioned)
- `baseline_metadata_<version>.json` (versioned)
- `current_model.json` (active version manifest)

Note: if training data has only one class, training falls back to a `DummyClassifier` until more varied data is collected.

API options:

- `POST /model/train`
- `GET /model/status`
- `GET /model/history?limit=20`
- `GET /predict?ticker=NVDA`
- `GET /prediction/logs?limit=100`
- `GET /docs/text`
- `POST /run/full?train_model=true`

Prediction behavior:
- If confidence is below `PREDICT_HOLD_THRESHOLD`, endpoint returns `prediction=hold`.
- All API predictions are logged in `prediction_logs`.

## 8) Runbook Commands (PowerShell)

- Full refresh + train:
  - `.\scripts\run_full_refresh.ps1`
- Seed missing article hashes for incremental NLP:
  - `.\scripts\seed_article_hashes.ps1`
- Seed NLP processed markers from current article hashes:
  - `.\scripts\seed_nlp_markers.ps1`
- Data status/quality report:
  - `.\scripts\show_data_checks.ps1`
- Local one-off prediction:
  - `.\scripts\predict_ticker.ps1 -Ticker SPY`

## Data model

- `news_articles`: raw article records + extracted tickers
  - retained in this repo for reference and future experiments, but not part of the active default pipeline
  - now also stores `source_type`, `external_id`, `import_batch`, `metadata_text` for historical imports
  - now also stores `source_hash`, `nlp_source_hash`, and `nlp_processed_at` so NLP can run incrementally
- `market_prices`: OHLCV snapshots per ticker/interval
- `ingestion_runs`: job-level audit trail
- `news_signals`: per article/ticker sentiment + relevance rows
- rerunning NLP now refreshes existing `news_signals` as well as inserting new ones
- broad macro and tech-heavy articles can now attach more aggressively to market proxy tickers like `SPY` and `QQQ`
- historical GKG rows now go through stricter noise filtering and minimum relevance thresholds before a ticker signal is kept
- NLP now processes `news_articles` in batches and only for rows that are new or whose source hash changed
- one-time maintenance helpers can seed `source_hash` and `nlp_processed_at` / `nlp_source_hash` on existing corpora
- `feature_snapshots`: rolling aggregates per ticker/window, including multi-horizon price returns, moving-average gaps, ATR/range, volatility regime, volume trend, plus retained news fields
- feature generation now refreshes existing snapshot rows when newer signals arrive inside their time window
- feature generation now also revisits stale snapshot rows that are missing newly added feature columns, so one-off feature upgrades can be backfilled without a full table rebuild
- feature generation now targets only market windows affected by recently NLP-processed articles instead of sweeping all windows on every run
- `market_labels`: horizon-based direction labels from future returns
- `prediction_logs`: API prediction audit history (ticker, prediction, confidence, model version, timestamp)
- `model_runs`: persisted retrain history used by the UI to compare current vs previous model metrics

## Current operating mode

This repo now runs in price-first mode by default:
- market ingestion stays active
- feature generation stays active
- model training uses `TRAINING_FEATURE_SET=price_only`
- retrains are stored in `model_runs` and shown in the dashboard automatically
- news ingestion/NLP can still be re-enabled later with `ENABLE_NEWS_PIPELINE=true`
