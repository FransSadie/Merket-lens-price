import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from app.core.config import get_settings
from app.data_quality.checks import data_status_snapshot, run_data_quality_checks
from app.db.models import (
    FeatureSnapshot,
    IngestionRun,
    MarketLabel,
    MarketPrice,
    ModelRun,
    NewsArticle,
    NewsSignal,
    PredictionLog,
)
from app.db.session import get_db_session
from app.features.pipeline import run_feature_generation, run_label_generation
from app.ingestion.gdelt_gkg_normalizer import normalize_gdelt_gkg_batch
from app.ingestion.historical_news import run_historical_news_import
from app.ingestion.pipeline import run_active_ingestion
from app.models.inference import log_prediction, predict_for_ticker
from app.models.train_baseline import load_experiment_matrix, run_experiment_matrix, train_and_save_baseline
from app.nlp.seed_article_hashes import seed_article_hashes
from app.nlp.seed_nlp_markers import seed_nlp_markers
from app.nlp.pipeline import run_news_nlp

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/ingest/run")
def ingest_run() -> dict:
    return run_active_ingestion()


@router.post("/ingest/historical/run")
def ingest_historical_run() -> dict:
    return run_historical_news_import()


@router.post("/normalize/gdelt/run")
def normalize_gdelt_run() -> dict:
    return normalize_gdelt_gkg_batch()


@router.get("/ingest/status")
def ingest_status() -> dict:
    session = get_db_session()
    try:
        latest_run = session.execute(select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(1)).scalar_one_or_none()
        article_count = session.execute(select(func.count(NewsArticle.id))).scalar_one()
        price_count = session.execute(select(func.count(MarketPrice.id))).scalar_one()
        return {
            "latest_run": {
                "job_name": latest_run.job_name if latest_run else None,
                "status": latest_run.status if latest_run else None,
                "rows_inserted": latest_run.rows_inserted if latest_run else None,
                "started_at": latest_run.started_at.isoformat() if latest_run else None,
            },
            "news_enabled": get_settings().enable_news_pipeline,
            "totals": {"news_articles": article_count, "market_prices": price_count},
        }
    finally:
        session.close()


@router.post("/pipeline/run")
def pipeline_run() -> dict:
    settings = get_settings()
    news_signal_count = run_news_nlp() if settings.enable_news_pipeline else 0
    feature_count = run_feature_generation(window_hours=24)
    label_count = run_label_generation(horizon_days=settings.training_horizon_days)
    return {
        "news_enabled": settings.enable_news_pipeline,
        "news_signals_inserted": news_signal_count,
        "feature_rows_inserted": feature_count,
        "labels_inserted": label_count,
    }


@router.post("/maintenance/seed-article-hashes")
def maintenance_seed_article_hashes() -> dict:
    return seed_article_hashes()


@router.post("/maintenance/seed-nlp-markers")
def maintenance_seed_nlp_markers() -> dict:
    return seed_nlp_markers()


@router.post("/run/full")
def run_full(train_model: bool = True) -> dict:
    settings = get_settings()
    ingestion = run_active_ingestion()
    news_signal_count = run_news_nlp() if settings.enable_news_pipeline else 0
    feature_count = run_feature_generation(window_hours=24)
    label_count = run_label_generation(horizon_days=settings.training_horizon_days)
    model_result = None
    if train_model:
        model_result = train_and_save_baseline()
    return {
        "ingestion": ingestion,
        "pipeline": {
            "news_enabled": settings.enable_news_pipeline,
            "news_signals_inserted": news_signal_count,
            "feature_rows_inserted": feature_count,
            "labels_inserted": label_count,
        },
        "model_training": model_result,
    }


@router.get("/pipeline/status")
def pipeline_status() -> dict:
    session = get_db_session()
    try:
        signal_count = session.execute(select(func.count(NewsSignal.id))).scalar_one()
        feature_count = session.execute(select(func.count(FeatureSnapshot.id))).scalar_one()
        label_count = session.execute(select(func.count(MarketLabel.id))).scalar_one()
        latest_run = session.execute(select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(1)).scalar_one_or_none()
        return {
            "latest_run": {
                "job_name": latest_run.job_name if latest_run else None,
                "status": latest_run.status if latest_run else None,
                "rows_inserted": latest_run.rows_inserted if latest_run else None,
                "started_at": latest_run.started_at.isoformat() if latest_run else None,
            },
            "news_enabled": get_settings().enable_news_pipeline,
            "totals": {
                "news_signals": signal_count,
                "feature_snapshots": feature_count,
                "market_labels": label_count,
            },
        }
    finally:
        session.close()


@router.post("/model/train")
def model_train() -> dict:
    try:
        return train_and_save_baseline()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/model/status")
def model_status() -> dict:
    settings = get_settings()
    metadata_path = Path(settings.model_artifacts_dir) / "baseline_metadata.json"
    if not metadata_path.exists():
        return {"trained": False, "metadata": None}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"trained": True, "metadata": metadata}


@router.get("/model/history")
def model_history(limit: int = 20) -> dict:
    page_size = max(1, min(limit, 200))
    session = get_db_session()
    try:
        rows = session.execute(
            select(ModelRun).order_by(ModelRun.created_at.desc()).limit(page_size)
        ).scalars().all()
        return {
            "rows": [
                {
                    "version_id": row.version_id,
                    "model_type": row.model_type,
                    "training_feature_set": row.training_feature_set,
                    "horizon_days": row.horizon_days,
                    "target_return_threshold": row.target_return_threshold,
                    "accuracy": row.accuracy,
                    "precision": row.precision,
                    "recall": row.recall,
                    "f1": row.f1,
                    "roc_auc": row.roc_auc,
                    "walk_forward_accuracy": row.walk_forward_accuracy,
                    "walk_forward_f1": row.walk_forward_f1,
                    "signals_count": row.signals_count,
                    "signals_hit_rate": row.signals_hit_rate,
                    "signals_avg_future_return": row.signals_avg_future_return,
                    "recommended_hold_threshold": row.recommended_hold_threshold,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        }
    finally:
        session.close()




@router.post("/model/experiments/run")
def model_experiments_run() -> dict:
    try:
        return run_experiment_matrix()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/model/experiments/status")
def model_experiments_status() -> dict:
    return load_experiment_matrix()


@router.get("/predict")
def predict(ticker: str) -> dict:
    try:
        result = predict_for_ticker(ticker=ticker)
        log_prediction(result)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/data/status")
def data_status() -> dict:
    return data_status_snapshot()


@router.get("/data/quality")
def data_quality() -> dict:
    return run_data_quality_checks()


@router.get("/prediction/logs")
def prediction_logs(limit: int = 100) -> dict:
    page_size = max(1, min(limit, 1000))
    session = get_db_session()
    try:
        rows = session.execute(
            select(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(page_size)
        ).scalars().all()
        return {
            "rows": [
                {
                    "id": row.id,
                    "ticker": row.ticker,
                    "prediction": row.prediction,
                    "probability_up": row.probability_up,
                    "confidence": row.confidence,
                    "model_version": row.model_version,
                    "window_end": row.window_end.isoformat() if row.window_end else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        }
    finally:
        session.close()


@router.get("/docs/text")
def docs_text() -> dict:
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    overview = (root / "PROJECT_OVERVIEW.txt").read_text(encoding="utf-8")
    return {"readme_markdown": readme, "project_overview_text": overview}
