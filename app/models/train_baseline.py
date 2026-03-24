import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.db.models import ModelRun
from app.db.session import get_db_session
from app.models.dataset import FEATURE_COLUMNS, load_training_dataframe, split_time_ordered

PRICE_ONLY_COLUMNS = [
    "price_close",
    "return_1d",
    "price_return_3d",
    "price_return_5d",
    "price_return_10d",
    "price_return_20d",
    "ma_gap_5d",
    "ma_gap_20d",
    "ma_crossover_5_20",
    "range_pct_1d",
    "atr_14_pct",
    "rolling_volatility_20d",
    "volatility_regime_60d",
    "volume_zscore_20d",
    "volume_change_5d",
]


def _selected_feature_columns(settings) -> list[str]:
    feature_set = (settings.training_feature_set or "price_only").strip().lower()
    if feature_set == "all":
        return FEATURE_COLUMNS
    return PRICE_ONLY_COLUMNS


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _probability_up(model: object, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(shape=(x.shape[0],), dtype=float)


def _build_model(y_train: np.ndarray) -> tuple[object, str]:
    train_classes = np.unique(y_train)
    if len(train_classes) < 2:
        return DummyClassifier(strategy="most_frequent"), "dummy_most_frequent"
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42, C=0.7),
            ),
        ]
    )
    return model, "scaled_logistic_regression"


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_proba),
    }


def _walk_forward_metrics(df, feature_cols: list[str], folds: int = 4) -> dict:
    if len(df) < 20:
        return {"folds": 0, "avg_accuracy": None, "avg_f1": None}

    fold_metrics = []
    step = max(5, len(df) // (folds + 1))
    for end_idx in range(step, len(df) - step + 1, step):
        train_df = df.iloc[:end_idx]
        val_df = df.iloc[end_idx : end_idx + step]
        if len(val_df) == 0:
            continue
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["target_up"].to_numpy(dtype=int)
        x_val = val_df[feature_cols].to_numpy(dtype=float)
        y_val = val_df["target_up"].to_numpy(dtype=int)

        model, _ = _build_model(y_train)
        model.fit(x_train, y_train)
        proba = _probability_up(model, x_val)
        pred = (proba >= 0.5).astype(int)
        fold_metrics.append(_classification_metrics(y_val, pred, proba))

    if not fold_metrics:
        return {"folds": 0, "avg_accuracy": None, "avg_f1": None}
    return {
        "folds": len(fold_metrics),
        "avg_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "avg_f1": float(np.mean([m["f1"] for m in fold_metrics])),
    }


def _strategy_summary(probabilities: np.ndarray, future_returns: np.ndarray, threshold: float) -> dict:
    mask = probabilities >= threshold
    selected = future_returns[mask]
    if selected.size == 0:
        return {
            "threshold": float(threshold),
            "signals_count": 0,
            "hit_rate": None,
            "avg_future_return": None,
            "compound_return": None,
            "max_drawdown": None,
        }
    equity = np.cumprod(1.0 + selected)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return {
        "threshold": float(threshold),
        "signals_count": int(selected.size),
        "hit_rate": float((selected > 0).mean()),
        "avg_future_return": float(selected.mean()),
        "compound_return": float(equity[-1] - 1.0),
        "max_drawdown": float(drawdowns.min()) if drawdowns.size else None,
    }


def _threshold_analysis(probabilities: np.ndarray, future_returns: np.ndarray) -> tuple[list[dict], float | None]:
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    rows = [_strategy_summary(probabilities, future_returns, threshold) for threshold in thresholds]
    min_signals = max(5, int(len(future_returns) * 0.05))
    eligible = [row for row in rows if row["signals_count"] >= min_signals and row["avg_future_return"] is not None]
    if not eligible:
        eligible = [row for row in rows if row["signals_count"] > 0 and row["avg_future_return"] is not None]
    if not eligible:
        return rows, None
    best = max(eligible, key=lambda row: (row["avg_future_return"], row["hit_rate"] or 0.0))
    return rows, float(best["threshold"])


def _baseline_comparison(train_df, val_df, active_feature_columns: list[str], active_model_label: str) -> dict:
    y_train = train_df["target_up"].to_numpy(dtype=int)
    y_val = val_df["target_up"].to_numpy(dtype=int)
    comparisons = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[active_feature_columns].to_numpy(dtype=float), y_train)
    dummy_proba = _probability_up(dummy, val_df[active_feature_columns].to_numpy(dtype=float))
    dummy_pred = (dummy_proba >= 0.5).astype(int)
    comparisons["dummy_most_frequent"] = _classification_metrics(y_val, dummy_pred, dummy_proba)

    price_model, price_model_type = _build_model(y_train)
    price_model.fit(train_df[PRICE_ONLY_COLUMNS].to_numpy(dtype=float), y_train)
    price_proba = _probability_up(price_model, val_df[PRICE_ONLY_COLUMNS].to_numpy(dtype=float))
    price_pred = (price_proba >= 0.5).astype(int)
    comparisons[f"price_only_{price_model_type}"] = _classification_metrics(y_val, price_pred, price_proba)

    if active_feature_columns != FEATURE_COLUMNS:
        full_model, full_model_type = _build_model(y_train)
        full_model.fit(train_df[FEATURE_COLUMNS].to_numpy(dtype=float), y_train)
        full_proba = _probability_up(full_model, val_df[FEATURE_COLUMNS].to_numpy(dtype=float))
        full_pred = (full_proba >= 0.5).astype(int)
        comparisons[f"full_feature_{full_model_type}"] = _classification_metrics(y_val, full_pred, full_proba)

    comparisons["active_model_feature_set"] = {"name": active_model_label}
    return comparisons


def _save_model_run(version_id: str, model_type: str, feature_set_name: str, horizon_days: int, target_threshold: float, metrics: dict, recommended_hold_threshold: float | None) -> None:
    session = get_db_session()
    try:
        row = ModelRun(
            version_id=version_id,
            model_type=model_type,
            training_feature_set=feature_set_name,
            horizon_days=horizon_days,
            target_return_threshold=float(target_threshold),
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1=metrics.get("f1"),
            roc_auc=metrics.get("roc_auc"),
            walk_forward_accuracy=(metrics.get("walk_forward") or {}).get("avg_accuracy"),
            walk_forward_f1=(metrics.get("walk_forward") or {}).get("avg_f1"),
            signals_count=metrics.get("signals_count"),
            signals_hit_rate=metrics.get("signals_hit_rate"),
            signals_avg_future_return=metrics.get("signals_avg_future_return"),
            recommended_hold_threshold=recommended_hold_threshold,
            metrics_json=json.dumps(metrics),
        )
        session.add(row)
        session.commit()
    finally:
        session.close()


def _seed_model_runs_from_artifacts(artifacts_dir: Path) -> None:
    session = get_db_session()
    try:
        known = {
            row[0]
            for row in session.query(ModelRun.version_id).all()
        }
        for metadata_path in sorted(artifacts_dir.glob("baseline_metadata_*.json")):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            version_id = payload.get("version_id")
            if not version_id or version_id in known:
                continue
            metrics = payload.get("metrics") or {}
            row = ModelRun(
                version_id=version_id,
                model_type=payload.get("model_type", "unknown"),
                training_feature_set=payload.get("training_feature_set", "unknown"),
                horizon_days=int(payload.get("horizon_days", 1)),
                target_return_threshold=float(payload.get("target_return_threshold", 0.0)),
                accuracy=metrics.get("accuracy"),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1=metrics.get("f1"),
                roc_auc=metrics.get("roc_auc"),
                walk_forward_accuracy=(metrics.get("walk_forward") or {}).get("avg_accuracy"),
                walk_forward_f1=(metrics.get("walk_forward") or {}).get("avg_f1"),
                signals_count=metrics.get("signals_count"),
                signals_hit_rate=metrics.get("signals_hit_rate"),
                signals_avg_future_return=metrics.get("signals_avg_future_return"),
                recommended_hold_threshold=metrics.get("recommended_hold_threshold"),
                metrics_json=json.dumps(metrics),
                created_at=datetime.fromisoformat(payload.get("created_at").replace("Z", "+00:00")).replace(tzinfo=None) if payload.get("created_at") else datetime.now(timezone.utc).replace(tzinfo=None),
            )
            session.add(row)
            known.add(version_id)
        session.commit()
    finally:
        session.close()


def train_and_save_baseline() -> dict:
    settings = get_settings()
    artifacts_dir = Path(settings.model_artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _seed_model_runs_from_artifacts(artifacts_dir)

    feature_columns = _selected_feature_columns(settings)
    feature_set_name = (settings.training_feature_set or "price_only").strip().lower()
    horizon_days = int(settings.training_horizon_days)
    target_threshold = float(settings.training_target_return_threshold)

    df = load_training_dataframe(
        horizon_days=horizon_days,
        window_hours=24,
        target_return_threshold=target_threshold,
    )
    if df.empty:
        raise RuntimeError("No training rows available. Run ingestion and feature pipeline first.")
    if len(df) < 3:
        raise RuntimeError(f"Not enough rows to train reliably (found {len(df)}).")

    train_df, val_df = split_time_ordered(df, train_ratio=0.8)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Time split failed due to insufficient rows.")

    x_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["target_up"].to_numpy(dtype=int)
    x_val = val_df[feature_columns].to_numpy(dtype=float)
    y_val = val_df["target_up"].to_numpy(dtype=int)

    train_classes = np.unique(y_train)
    model, model_type = _build_model(y_train)
    model.fit(x_train, y_train)

    val_proba = _probability_up(model, x_val)
    val_pred = (val_proba >= 0.5).astype(int)

    selected_returns = val_df.loc[val_pred == 1, "future_return"]
    hit_rate = float((selected_returns > 0).mean()) if len(selected_returns) > 0 else None
    avg_future_return = float(selected_returns.mean()) if len(selected_returns) > 0 else None
    threshold_rows, recommended_hold_threshold = _threshold_analysis(
        val_proba,
        val_df["future_return"].to_numpy(dtype=float),
    )
    default_strategy = _strategy_summary(
        val_proba,
        val_df["future_return"].to_numpy(dtype=float),
        settings.predict_hold_threshold,
    )

    metrics = {
        **_classification_metrics(y_val, val_pred, val_proba),
        "val_rows": int(len(val_df)),
        "train_rows": int(len(train_df)),
        "signals_count": int((val_pred == 1).sum()),
        "signals_hit_rate": hit_rate,
        "signals_avg_future_return": avg_future_return,
        "walk_forward": _walk_forward_metrics(df, feature_columns, folds=4),
        "baselines": _baseline_comparison(train_df, val_df, feature_columns, feature_set_name),
        "threshold_analysis": threshold_rows,
        "recommended_hold_threshold": recommended_hold_threshold,
        "default_hold_threshold_summary": default_strategy,
    }

    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = artifacts_dir / f"baseline_model_{version_id}.joblib"
    metadata_path = artifacts_dir / f"baseline_metadata_{version_id}.json"
    latest_model_path = artifacts_dir / "baseline_model.joblib"
    latest_metadata_path = artifacts_dir / "baseline_metadata.json"
    manifest_path = artifacts_dir / "current_model.json"

    joblib.dump(model, model_path)
    joblib.dump(model, latest_model_path)
    metadata = {
        "version_id": version_id,
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns,
        "training_feature_set": feature_set_name,
        "horizon_days": horizon_days,
        "window_hours": 24,
        "train_classes": [int(v) for v in train_classes.tolist()],
        "target_return_threshold": target_threshold,
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    latest_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "version_id": version_id,
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_model_run(version_id, model_type, feature_set_name, horizon_days, target_threshold, metrics, recommended_hold_threshold)

    return {
        "version_id": version_id,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
    }


def main() -> None:
    result = train_and_save_baseline()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
