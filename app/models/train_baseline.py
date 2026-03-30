import json
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.db.models import ModelRun
from app.db.session import get_db_session
from app.features.pipeline import run_label_generation
from app.models.dataset import FEATURE_COLUMNS, load_training_dataframe, split_time_ordered

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

PRICE_ONLY_COLUMNS = [
    "price_close",
    "return_1d",
    "price_return_2d",
    "price_return_3d",
    "price_return_5d",
    "price_return_10d",
    "price_return_15d",
    "price_return_20d",
    "ma_gap_5d",
    "ma_gap_20d",
    "ma_crossover_5_20",
    "ema_gap_12d",
    "ema_gap_26d",
    "ema_crossover_12_26",
    "range_pct_1d",
    "atr_14_pct",
    "rolling_volatility_20d",
    "downside_volatility_20d",
    "volatility_regime_60d",
    "volume_zscore_20d",
    "volume_change_5d",
    "volume_ratio_20d",
    "rsi_5",
    "rsi_14",
    "bollinger_z_20",
    "breakout_20d",
    "drawdown_20d",
    "trend_slope_10d",
    "trend_slope_20d",
    "up_day_ratio_10d",
    "rel_to_spy_return_5d",
    "rel_to_spy_return_20d",
    "rel_to_qqq_return_5d",
    "rel_to_sector_return_5d",
    "rel_to_sector_return_20d",
    "market_beta_20d",
    "market_corr_20d",
    "sector_corr_20d",
    "volume_vs_spy_ratio_20d",
]

DEFAULT_EXPERIMENT_HORIZONS = [1, 3, 5]
DEFAULT_EXPERIMENT_THRESHOLDS = [0.002, 0.004, 0.006]
DEFAULT_MAX_POSITIONS_PER_DAY = 5


def _selected_feature_columns(settings) -> list[str]:
    feature_set = (settings.training_feature_set or "price_only").strip().lower()
    if feature_set == "all":
        return FEATURE_COLUMNS
    return PRICE_ONLY_COLUMNS


def _base_training_dataframe(horizon_days: int) -> pd.DataFrame:
    return load_training_dataframe(
        horizon_days=horizon_days,
        window_hours=24,
        target_return_threshold=0.0,
    )


def _prepare_training_dataframe(base_df: pd.DataFrame, target_return_threshold: float) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame()
    df = base_df.copy()
    df["target_up"] = (df["future_return"] > float(target_return_threshold)).astype(int)
    return df


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _probability_up(model: object, x) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(shape=(len(x),), dtype=float)


def _build_dummy_model() -> tuple[object, str]:
    return DummyClassifier(strategy="most_frequent"), "dummy_most_frequent"


def _build_logistic_model(y_train: np.ndarray) -> tuple[object, str]:
    train_classes = np.unique(y_train)
    if len(train_classes) < 2:
        return _build_dummy_model()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, C=0.7)),
        ]
    )
    return model, "scaled_logistic_regression"


def _lightgbm_base_params(settings) -> dict:
    return {
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": 42,
        "verbosity": -1,
        "num_leaves": int(settings.lightgbm_num_leaves),
        "learning_rate": float(settings.lightgbm_learning_rate),
        "n_estimators": int(settings.lightgbm_n_estimators),
        "max_depth": int(settings.lightgbm_max_depth),
        "min_child_samples": int(settings.lightgbm_min_child_samples),
        "subsample": float(settings.lightgbm_subsample),
        "colsample_bytree": float(settings.lightgbm_colsample_bytree),
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
    }


def _build_lightgbm_model(settings, y_train: np.ndarray, params: dict | None = None) -> tuple[object, str]:
    train_classes = np.unique(y_train)
    if len(train_classes) < 2:
        return _build_dummy_model()
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is not installed. Run `pip install -r requirements.txt` first.")
    model = LGBMClassifier(**(params or _lightgbm_base_params(settings)))
    return model, "lightgbm"


def _build_active_model(settings, y_train: np.ndarray, params: dict | None = None) -> tuple[object, str]:
    model_type = (settings.model_type or "lightgbm").strip().lower()
    if model_type == "logistic_regression":
        return _build_logistic_model(y_train)
    if model_type == "lightgbm":
        return _build_lightgbm_model(settings, y_train, params=params)
    raise RuntimeError(f"Unsupported MODEL_TYPE={settings.model_type}")


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_proba),
    }


def _strategy_trades(probabilities: np.ndarray, strategy_df: pd.DataFrame, threshold: float, horizon_days: int, max_positions_per_day: int = DEFAULT_MAX_POSITIONS_PER_DAY) -> pd.DataFrame:
    trades = strategy_df.copy()
    trades["probability_up"] = probabilities
    trades = trades[trades["probability_up"] >= threshold].copy()
    if trades.empty:
        return trades

    trades["window_end"] = pd.to_datetime(trades["window_end"])
    trades["trade_date"] = trades["window_end"].dt.normalize()
    trades = trades.sort_values(by=["trade_date", "probability_up", "ticker"], ascending=[True, False, True])

    selected_rows = []
    last_trade_date_by_ticker: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(days=max(1, int(horizon_days)))

    for trade_date, day_rows in trades.groupby("trade_date", sort=True):
        chosen_for_day = 0
        for _, row in day_rows.iterrows():
            last_trade_date = last_trade_date_by_ticker.get(row["ticker"])
            if last_trade_date is not None and trade_date < last_trade_date + cooldown:
                continue
            selected_rows.append(row.to_dict())
            last_trade_date_by_ticker[row["ticker"]] = trade_date
            chosen_for_day += 1
            if chosen_for_day >= max_positions_per_day:
                break

    if not selected_rows:
        return pd.DataFrame(columns=list(trades.columns))
    return pd.DataFrame(selected_rows)


def _strategy_summary(probabilities: np.ndarray, strategy_df: pd.DataFrame, threshold: float, horizon_days: int, max_positions_per_day: int = DEFAULT_MAX_POSITIONS_PER_DAY) -> dict:
    selected = _strategy_trades(probabilities, strategy_df, threshold, horizon_days, max_positions_per_day=max_positions_per_day)
    if selected.empty:
        return {
            "threshold": float(threshold),
            "signals_count": 0,
            "trade_count": 0,
            "active_days": 0,
            "hit_rate": None,
            "avg_future_return": None,
            "avg_daily_return": None,
            "compound_return": None,
            "max_drawdown": None,
            "max_positions_per_day": int(max_positions_per_day),
            "horizon_days": int(horizon_days),
        }

    selected_returns = selected["future_return"].astype(float).to_numpy()
    daily_returns = (
        selected.groupby("trade_date", sort=True)["future_return"]
        .mean()
        .astype(float)
        .to_numpy()
    )
    equity = np.cumprod(1.0 + daily_returns)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0

    return {
        "threshold": float(threshold),
        "signals_count": int(selected.shape[0]),
        "trade_count": int(selected.shape[0]),
        "active_days": int(len(daily_returns)),
        "hit_rate": float((selected_returns > 0).mean()),
        "avg_future_return": float(selected_returns.mean()),
        "avg_daily_return": float(daily_returns.mean()),
        "compound_return": float(equity[-1] - 1.0),
        "max_drawdown": float(drawdowns.min()) if drawdowns.size else None,
        "max_positions_per_day": int(max_positions_per_day),
        "horizon_days": int(horizon_days),
    }


def _threshold_analysis(probabilities: np.ndarray, strategy_df: pd.DataFrame, horizon_days: int) -> tuple[list[dict], float | None]:
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    rows = [_strategy_summary(probabilities, strategy_df, threshold, horizon_days) for threshold in thresholds]
    min_signals = max(10, int(len(strategy_df) * 0.02))
    eligible = [row for row in rows if row["signals_count"] >= min_signals and row["avg_daily_return"] is not None]
    if not eligible:
        eligible = [row for row in rows if row["signals_count"] > 0 and row["avg_daily_return"] is not None]
    if not eligible:
        return rows, None
    best = max(
        eligible,
        key=lambda row: (
            row["avg_daily_return"],
            row["hit_rate"] or 0.0,
            -abs(row["max_drawdown"] or 0.0),
        ),
    )
    return rows, float(best["threshold"])


def _candidate_score(metrics: dict, strategy: dict) -> float:
    roc_auc = metrics.get("roc_auc") or 0.0
    f1 = metrics.get("f1") or 0.0
    accuracy = metrics.get("accuracy") or 0.0
    avg_daily_return = strategy.get("avg_daily_return") or 0.0
    hit_rate = strategy.get("hit_rate") or 0.0
    max_drawdown = abs(strategy.get("max_drawdown") or 0.0)
    return (roc_auc * 4.0) + (f1 * 3.0) + (accuracy * 1.5) + (hit_rate * 1.0) + (avg_daily_return * 250.0) - (max_drawdown * 1.5)


def _lightgbm_candidate_params(settings) -> list[dict]:
    base = _lightgbm_base_params(settings)
    variants = [
        {},
        {"num_leaves": 7, "max_depth": 4, "min_child_samples": max(30, int(settings.lightgbm_min_child_samples))},
        {"num_leaves": 15, "max_depth": 5, "min_child_samples": max(24, int(settings.lightgbm_min_child_samples))},
        {"num_leaves": 31, "max_depth": 6, "min_child_samples": max(16, int(settings.lightgbm_min_child_samples) - 4)},
        {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.03, "n_estimators": max(300, int(settings.lightgbm_n_estimators) + 100)},
        {"num_leaves": 15, "max_depth": 3, "learning_rate": 0.08, "n_estimators": 120},
    ]
    candidates = []
    seen = set()
    for overrides in variants:
        params = deepcopy(base)
        params.update(overrides)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(params)
    return candidates


def _evaluate_lightgbm_candidates(settings, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], horizon_days: int) -> tuple[object, dict, list[dict], np.ndarray]:
    y_train = train_df["target_up"].to_numpy(dtype=int)
    x_train = train_df[feature_cols].astype(float)
    x_val = val_df[feature_cols].astype(float)
    y_val = val_df["target_up"].to_numpy(dtype=int)
    strategy_df = val_df[["ticker", "window_end", "future_return"]].copy()

    trials = []
    best = None
    for idx, params in enumerate(_lightgbm_candidate_params(settings), start=1):
        model, model_type = _build_lightgbm_model(settings, y_train, params=params)
        model.fit(x_train, y_train)
        proba = _probability_up(model, x_val)
        pred = (proba >= 0.5).astype(int)
        metrics = _classification_metrics(y_val, pred, proba)
        threshold_rows, recommended_threshold = _threshold_analysis(proba, strategy_df, horizon_days)
        default_strategy = _strategy_summary(proba, strategy_df, settings.predict_hold_threshold, horizon_days)
        recommended_strategy = next((row for row in threshold_rows if row["threshold"] == recommended_threshold), None)
        score = _candidate_score(metrics, recommended_strategy or default_strategy)
        trial = {
            "trial": idx,
            "model_type": model_type,
            "params": params,
            "metrics": metrics,
            "recommended_hold_threshold": recommended_threshold,
            "default_hold_threshold_summary": default_strategy,
            "recommended_threshold_summary": recommended_strategy,
            "score": float(score),
        }
        trials.append(trial)
        if best is None or score > best["score"]:
            best = {"score": score, "model": model, "trial": trial, "proba": proba}

    return best["model"], best["trial"], trials, best["proba"]


def _walk_forward_metrics(df, feature_cols: list[str], settings, folds: int = 4) -> dict:
    if len(df) < 20:
        return {"folds": 0, "avg_accuracy": None, "avg_f1": None, "avg_roc_auc": None}

    fold_metrics = []
    step = max(5, len(df) // (folds + 1))
    for end_idx in range(step, len(df) - step + 1, step):
        train_df = df.iloc[:end_idx]
        val_df = df.iloc[end_idx : end_idx + step]
        if len(val_df) == 0:
            continue
        y_train = train_df["target_up"].to_numpy(dtype=int)
        x_train = train_df[feature_cols].astype(float)
        x_val = val_df[feature_cols].astype(float)
        y_val = val_df["target_up"].to_numpy(dtype=int)

        if (settings.model_type or "lightgbm").strip().lower() == "lightgbm":
            params = _lightgbm_candidate_params(settings)[0]
            model, _ = _build_lightgbm_model(settings, y_train, params=params)
        else:
            model, _ = _build_active_model(settings, y_train)
        model.fit(x_train, y_train)
        proba = _probability_up(model, x_val)
        pred = (proba >= 0.5).astype(int)
        fold_metrics.append(_classification_metrics(y_val, pred, proba))

    if not fold_metrics:
        return {"folds": 0, "avg_accuracy": None, "avg_f1": None, "avg_roc_auc": None}
    return {
        "folds": len(fold_metrics),
        "avg_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "avg_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "avg_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics if m["roc_auc"] is not None])) if any(m["roc_auc"] is not None for m in fold_metrics) else None,
    }


def _baseline_comparison(train_df, val_df, active_feature_columns: list[str]) -> dict:
    y_train = train_df["target_up"].to_numpy(dtype=int)
    y_val = val_df["target_up"].to_numpy(dtype=int)
    comparisons = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[active_feature_columns].astype(float), y_train)
    dummy_proba = _probability_up(dummy, val_df[active_feature_columns].astype(float))
    dummy_pred = (dummy_proba >= 0.5).astype(int)
    comparisons["dummy_most_frequent"] = _classification_metrics(y_val, dummy_pred, dummy_proba)

    logistic_model, logistic_model_type = _build_logistic_model(y_train)
    logistic_model.fit(train_df[active_feature_columns].astype(float), y_train)
    logistic_proba = _probability_up(logistic_model, val_df[active_feature_columns].astype(float))
    logistic_pred = (logistic_proba >= 0.5).astype(int)
    comparisons[f"logistic_regression_{logistic_model_type}"] = _classification_metrics(y_val, logistic_pred, logistic_proba)

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
        known = {row[0] for row in session.query(ModelRun.version_id).all()}
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
                created_at=datetime.fromisoformat(payload.get("created_at").replace("Z", "+00:00")).replace(tzinfo=None)
                if payload.get("created_at")
                else datetime.now(timezone.utc).replace(tzinfo=None),
            )
            session.add(row)
            known.add(version_id)
        session.commit()
    finally:
        session.close()


def _experiment_matrix_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "experiment_matrix.json"


def load_experiment_matrix() -> dict:
    settings = get_settings()
    path = _experiment_matrix_path(Path(settings.model_artifacts_dir))
    if not path.exists():
        return {"available": False, "generated_at": None, "rows": [], "best_configuration": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["available"] = True
    return payload


def _evaluate_configuration(settings, feature_set_name: str, horizon_days: int, target_threshold: float, base_df: pd.DataFrame | None = None) -> dict:
    feature_columns = _selected_feature_columns(settings)
    raw_df = base_df if base_df is not None else _base_training_dataframe(horizon_days)
    df = _prepare_training_dataframe(raw_df, target_threshold)
    if df.empty:
        raise RuntimeError("No training rows available. Run ingestion and feature pipeline first.")
    if len(df) < 3:
        raise RuntimeError(f"Not enough rows to train reliably (found {len(df)}).")

    train_df, val_df = split_time_ordered(df, train_ratio=0.8)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Time split failed due to insufficient rows.")

    y_train = train_df["target_up"].to_numpy(dtype=int)
    y_val = val_df["target_up"].to_numpy(dtype=int)
    train_classes = np.unique(y_train)
    strategy_df = val_df[["ticker", "window_end", "future_return"]].copy()

    if (settings.model_type or "lightgbm").strip().lower() == "lightgbm":
        model, selected_trial, lightgbm_trials, val_proba = _evaluate_lightgbm_candidates(settings, train_df, val_df, feature_columns, horizon_days)
        model_type = "lightgbm"
        model_selection = {
            "selected_trial": selected_trial,
            "candidate_trials": lightgbm_trials,
        }
    else:
        model, model_type = _build_active_model(settings, y_train)
        model.fit(train_df[feature_columns].astype(float), y_train)
        val_proba = _probability_up(model, val_df[feature_columns].astype(float))
        model_selection = None

    val_pred = (val_proba >= 0.5).astype(int)
    active_metrics = _classification_metrics(y_val, val_pred, val_proba)
    threshold_rows, recommended_hold_threshold = _threshold_analysis(val_proba, strategy_df, horizon_days)
    default_strategy = _strategy_summary(val_proba, strategy_df, settings.predict_hold_threshold, horizon_days)
    recommended_strategy = next((row for row in threshold_rows if row["threshold"] == recommended_hold_threshold), None)

    benchmarks = _baseline_comparison(train_df, val_df, feature_columns)
    logistic_benchmark = next((value for key, value in benchmarks.items() if key.startswith("logistic_regression_")), None)
    benchmark_delta = None
    if logistic_benchmark:
        benchmark_delta = {
            "accuracy": float(active_metrics["accuracy"] - logistic_benchmark.get("accuracy", 0.0)),
            "f1": float(active_metrics["f1"] - logistic_benchmark.get("f1", 0.0)),
            "roc_auc": None if logistic_benchmark.get("roc_auc") is None or active_metrics["roc_auc"] is None else float(active_metrics["roc_auc"] - logistic_benchmark.get("roc_auc", 0.0)),
        }

    importance_rows = None
    if model_type == "lightgbm" and hasattr(model, "feature_importances_"):
        importance_rows = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(zip(feature_columns, model.feature_importances_), key=lambda item: item[1], reverse=True)
            if float(importance) > 0
        ][:15]

    metrics = {
        **active_metrics,
        "val_rows": int(len(val_df)),
        "train_rows": int(len(train_df)),
        "signals_count": int((val_pred == 1).sum()),
        "signals_hit_rate": default_strategy.get("hit_rate"),
        "signals_avg_future_return": default_strategy.get("avg_future_return"),
        "signals_avg_daily_return": default_strategy.get("avg_daily_return"),
        "walk_forward": _walk_forward_metrics(df, feature_columns, settings, folds=4),
        "benchmarks": benchmarks,
        "benchmark_comparison": {
            "active_model": model_type,
            "logistic_regression_delta": benchmark_delta,
        },
        "threshold_analysis": threshold_rows,
        "recommended_hold_threshold": recommended_hold_threshold,
        "default_hold_threshold_summary": default_strategy,
        "recommended_hold_threshold_summary": recommended_strategy,
        "model_selection": model_selection,
        "feature_importance_top": importance_rows,
        "strategy_constraints": {
            "max_positions_per_day": DEFAULT_MAX_POSITIONS_PER_DAY,
            "cooldown_horizon_days": int(horizon_days),
            "note": "Backtest summaries use capped daily positions and per-ticker cooldowns. Treat them as diagnostics, not broker-grade PnL.",
        },
    }

    return {
        "model": model,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "feature_set_name": feature_set_name,
        "horizon_days": int(horizon_days),
        "target_threshold": float(target_threshold),
        "train_classes": train_classes,
        "metrics": metrics,
    }


def run_experiment_matrix(horizons: list[int] | None = None, thresholds: list[float] | None = None) -> dict:
    settings = get_settings()
    artifacts_dir = Path(settings.model_artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    horizons = [int(v) for v in (horizons or DEFAULT_EXPERIMENT_HORIZONS)]
    thresholds = [float(v) for v in (thresholds or DEFAULT_EXPERIMENT_THRESHOLDS)]
    feature_set_name = (settings.training_feature_set or "price_only").strip().lower()

    rows = []
    base_datasets: dict[int, pd.DataFrame] = {}
    for horizon_days in horizons:
        run_label_generation(horizon_days=int(horizon_days))
        base_datasets[int(horizon_days)] = _base_training_dataframe(int(horizon_days))

    for horizon_days, target_threshold in product(horizons, thresholds):
        experiment_settings = settings.model_copy(
            update={
                "training_horizon_days": int(horizon_days),
                "training_target_return_threshold": float(target_threshold),
            }
        )
        result = _evaluate_configuration(
            experiment_settings,
            feature_set_name,
            horizon_days,
            target_threshold,
            base_df=base_datasets[int(horizon_days)],
        )
        metrics = result["metrics"]
        logistic_metrics = next(
            (value for key, value in (metrics.get("benchmarks") or {}).items() if key.startswith("logistic_regression_")),
            None,
        )
        row = {
            "model_type": result["model_type"],
            "training_feature_set": feature_set_name,
            "horizon_days": int(horizon_days),
            "target_return_threshold": float(target_threshold),
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "walk_forward_accuracy": (metrics.get("walk_forward") or {}).get("avg_accuracy"),
            "walk_forward_f1": (metrics.get("walk_forward") or {}).get("avg_f1"),
            "recommended_hold_threshold": metrics.get("recommended_hold_threshold"),
            "signals_count": metrics.get("signals_count"),
            "signals_hit_rate": metrics.get("signals_hit_rate"),
            "signals_avg_future_return": metrics.get("signals_avg_future_return"),
            "signals_avg_daily_return": metrics.get("signals_avg_daily_return"),
            "recommended_strategy": metrics.get("recommended_hold_threshold_summary"),
            "default_strategy": metrics.get("default_hold_threshold_summary"),
            "logistic_benchmark": logistic_metrics,
            "benchmark_delta": (metrics.get("benchmark_comparison") or {}).get("logistic_regression_delta"),
            "selected_trial": (metrics.get("model_selection") or {}).get("selected_trial"),
            "ranking_score": _candidate_score(
                {"accuracy": metrics.get("accuracy") or 0.0, "f1": metrics.get("f1") or 0.0, "roc_auc": metrics.get("roc_auc") or 0.0},
                metrics.get("recommended_hold_threshold_summary") or metrics.get("default_hold_threshold_summary") or {},
            ),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["ranking_score"],
            row.get("walk_forward_accuracy") or 0.0,
            row.get("roc_auc") or 0.0,
        ),
        reverse=True,
    )
    best_configuration = rows[0] if rows else None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_type": (settings.model_type or "lightgbm").strip().lower(),
        "training_feature_set": feature_set_name,
        "horizons": horizons,
        "thresholds": thresholds,
        "best_configuration": best_configuration,
        "rows": rows,
        "notes": [
            "Rows are ranked with a mix of ROC AUC, F1, accuracy, hit rate, avg daily return, and drawdown-aware strategy score.",
            "Strategy summaries cap positions per day and apply per-ticker cooldowns to reduce unrealistic compounding.",
        ],
    }
    _experiment_matrix_path(artifacts_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_and_save_baseline() -> dict:
    settings = get_settings()
    artifacts_dir = Path(settings.model_artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _seed_model_runs_from_artifacts(artifacts_dir)

    feature_set_name = (settings.training_feature_set or "price_only").strip().lower()
    result = _evaluate_configuration(
        settings,
        feature_set_name=feature_set_name,
        horizon_days=int(settings.training_horizon_days),
        target_threshold=float(settings.training_target_return_threshold),
    )

    metrics = result["metrics"]
    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = artifacts_dir / f"baseline_model_{version_id}.joblib"
    metadata_path = artifacts_dir / f"baseline_metadata_{version_id}.json"
    latest_model_path = artifacts_dir / "baseline_model.joblib"
    latest_metadata_path = artifacts_dir / "baseline_metadata.json"
    manifest_path = artifacts_dir / "current_model.json"

    joblib.dump(result["model"], model_path)
    joblib.dump(result["model"], latest_model_path)
    metadata = {
        "version_id": version_id,
        "model_type": result["model_type"],
        "active_model_type": result["model_type"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": result["feature_columns"],
        "training_feature_set": result["feature_set_name"],
        "horizon_days": result["horizon_days"],
        "window_hours": 24,
        "train_classes": [int(v) for v in result["train_classes"].tolist()],
        "target_return_threshold": result["target_threshold"],
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
                "model_type": result["model_type"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_model_run(version_id, result["model_type"], result["feature_set_name"], result["horizon_days"], result["target_threshold"], metrics, metrics.get("recommended_hold_threshold"))

    return {
        "version_id": version_id,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
        "model_type": result["model_type"],
    }


def main() -> None:
    result = train_and_save_baseline()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
