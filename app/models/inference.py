import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.db.models import PredictionLog
from app.db.session import get_db_session
from app.models.dataset import latest_feature_row_for_ticker


def _artifacts_paths() -> tuple[Path, Path, Path]:
    settings = get_settings()
    artifacts_dir = Path(settings.model_artifacts_dir)
    return (
        artifacts_dir / "baseline_model.joblib",
        artifacts_dir / "baseline_metadata.json",
        artifacts_dir / "current_model.json",
    )


def load_model_and_metadata() -> tuple[object, dict]:
    model_path, metadata_path, manifest_path = _artifacts_paths()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_path = Path(manifest.get("model_path", model_path))
        metadata_path = Path(manifest.get("metadata_path", metadata_path))
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts not found. Train model first.")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def predict_for_ticker(ticker: str) -> dict:
    settings = get_settings()
    model, metadata = load_model_and_metadata()
    feature_columns = metadata.get("feature_columns", [])

    row = latest_feature_row_for_ticker(ticker=ticker.upper(), window_hours=metadata.get("window_hours", 24))
    if not row:
        raise ValueError(f"No feature snapshot found for ticker {ticker.upper()}.")

    vector = []
    for col in feature_columns:
        value = row.get(col)
        if value is None:
            value = 0.0
        vector.append(float(value))

    x = pd.DataFrame([vector], columns=feature_columns, dtype=float)
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    if 1 in classes:
        probability_up = float(proba[classes.index(1)])
    else:
        probability_up = 0.0
    direction = "up" if probability_up >= 0.5 else "down"
    confidence = probability_up if direction == "up" else 1.0 - probability_up
    prediction = direction if confidence >= settings.predict_hold_threshold else "hold"

    return {
        "ticker": ticker.upper(),
        "prediction": prediction,
        "direction": direction,
        "probability_up": probability_up,
        "confidence": confidence,
        "model_version": metadata.get("version_id"),
        "window_end": row.get("window_end"),
        "features": {col: row.get(col) for col in feature_columns},
    }


def log_prediction(result: dict) -> None:
    session = get_db_session()
    try:
        window_end = result.get("window_end")
        parsed_window_end = None
        if window_end:
            try:
                parsed_window_end = datetime.fromisoformat(window_end)
            except ValueError:
                parsed_window_end = None
        row = PredictionLog(
            ticker=result["ticker"],
            prediction=result["prediction"],
            probability_up=float(result["probability_up"]),
            confidence=float(result["confidence"]),
            model_version=result.get("model_version"),
            window_end=parsed_window_end,
        )
        session.add(row)
        session.commit()
    finally:
        session.close()
