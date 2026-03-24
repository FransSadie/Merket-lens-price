from datetime import datetime

import pandas as pd
from sqlalchemy import select

from app.db.models import FeatureSnapshot, MarketLabel
from app.db.session import get_db_session

FEATURE_COLUMNS = [
    "news_count",
    "news_count_change_24h",
    "sentiment_mean",
    "sentiment_sum",
    "sentiment_std",
    "sentiment_momentum_24h",
    "relevance_mean",
    "max_relevance",
    "source_weight_mean",
    "weighted_news_count",
    "weighted_sentiment_sum",
    "positive_news_ratio",
    "negative_news_ratio",
    "source_diversity",
    "news_count_72h",
    "event_intensity",
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


def load_training_dataframe(horizon_days: int = 1, window_hours: int = 24, target_return_threshold: float = 0.0) -> pd.DataFrame:
    session = get_db_session()
    try:
        feature_rows = session.execute(
            select(FeatureSnapshot).where(FeatureSnapshot.window_hours == window_hours)
        ).scalars().all()
        label_rows = session.execute(select(MarketLabel).where(MarketLabel.horizon_days == horizon_days)).scalars().all()
    finally:
        session.close()

    features = pd.DataFrame(
        [
            {
                "ticker": row.ticker,
                "window_end": row.window_end,
                "news_count": row.news_count,
                "sentiment_mean": row.sentiment_mean,
                "sentiment_sum": row.sentiment_sum,
                "sentiment_std": row.sentiment_std,
                "relevance_mean": row.relevance_mean,
                "max_relevance": row.max_relevance,
                "source_weight_mean": row.source_weight_mean,
                "weighted_news_count": row.weighted_news_count,
                "weighted_sentiment_sum": row.weighted_sentiment_sum,
                "positive_news_ratio": row.positive_news_ratio,
                "negative_news_ratio": row.negative_news_ratio,
                "source_diversity": row.source_diversity,
                "news_count_72h": row.news_count_72h,
                "event_intensity": row.event_intensity,
                "news_count_change_24h": row.news_count_change_24h,
                "sentiment_momentum_24h": row.sentiment_momentum_24h,
                "price_close": row.price_close,
                "return_1d": row.return_1d,
                "price_return_3d": row.price_return_3d,
                "price_return_5d": row.price_return_5d,
                "price_return_10d": row.price_return_10d,
                "price_return_20d": row.price_return_20d,
                "ma_gap_5d": row.ma_gap_5d,
                "ma_gap_20d": row.ma_gap_20d,
                "ma_crossover_5_20": row.ma_crossover_5_20,
                "range_pct_1d": row.range_pct_1d,
                "atr_14_pct": row.atr_14_pct,
                "rolling_volatility_20d": row.rolling_volatility_20d,
                "volatility_regime_60d": row.volatility_regime_60d,
                "volume_zscore_20d": row.volume_zscore_20d,
                "volume_change_5d": row.volume_change_5d,
            }
            for row in feature_rows
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "ticker": row.ticker,
                "timestamp": row.timestamp,
                "target_up": row.target_up,
                "future_return": row.future_return,
            }
            for row in label_rows
        ]
    )

    if features.empty or labels.empty:
        return pd.DataFrame()

    features["feature_date"] = pd.to_datetime(features["window_end"]).dt.normalize()
    labels["label_date"] = pd.to_datetime(labels["timestamp"]).dt.normalize()

    features = features.sort_values(by=["feature_date", "ticker"])
    labels = labels.sort_values(by=["label_date", "ticker"])
    merged = pd.merge_asof(
        features,
        labels,
        left_on="feature_date",
        right_on="label_date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    if merged.empty:
        return pd.DataFrame()
    merged = merged.dropna(subset=["timestamp", "future_return"])

    merged["window_end"] = pd.to_datetime(merged["window_end"])
    merged = merged.sort_values(by="window_end").reset_index(drop=True)
    merged["target_up"] = (merged["future_return"] > float(target_return_threshold)).astype(int)

    for col in FEATURE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.fillna(0.0)
    return merged


def split_time_ordered(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    cutoff = max(1, int(len(df) * train_ratio))
    if cutoff >= len(df):
        cutoff = len(df) - 1
    train_df = df.iloc[:cutoff].copy()
    val_df = df.iloc[cutoff:].copy()
    return train_df, val_df


def latest_feature_row_for_ticker(ticker: str, window_hours: int = 24) -> dict | None:
    session = get_db_session()
    try:
        row = session.execute(
            select(FeatureSnapshot)
            .where(FeatureSnapshot.ticker == ticker.upper(), FeatureSnapshot.window_hours == window_hours)
            .order_by(FeatureSnapshot.window_end.desc())
            .limit(1)
        ).scalar_one_or_none()
    finally:
        session.close()

    if not row:
        return None

    data = {
        "ticker": row.ticker,
        "window_end": row.window_end.isoformat() if isinstance(row.window_end, datetime) else None,
        "news_count": row.news_count,
        "sentiment_mean": row.sentiment_mean,
        "sentiment_sum": row.sentiment_sum,
        "sentiment_std": row.sentiment_std,
        "relevance_mean": row.relevance_mean,
        "max_relevance": row.max_relevance,
        "source_weight_mean": row.source_weight_mean,
        "weighted_news_count": row.weighted_news_count,
        "weighted_sentiment_sum": row.weighted_sentiment_sum,
        "positive_news_ratio": row.positive_news_ratio,
        "negative_news_ratio": row.negative_news_ratio,
        "source_diversity": row.source_diversity,
        "news_count_72h": row.news_count_72h,
        "event_intensity": row.event_intensity,
        "news_count_change_24h": row.news_count_change_24h,
        "sentiment_momentum_24h": row.sentiment_momentum_24h,
        "price_close": row.price_close,
        "return_1d": row.return_1d,
        "price_return_3d": row.price_return_3d,
        "price_return_5d": row.price_return_5d,
        "price_return_10d": row.price_return_10d,
        "price_return_20d": row.price_return_20d,
        "ma_gap_5d": row.ma_gap_5d,
        "ma_gap_20d": row.ma_gap_20d,
        "ma_crossover_5_20": row.ma_crossover_5_20,
        "range_pct_1d": row.range_pct_1d,
        "atr_14_pct": row.atr_14_pct,
        "rolling_volatility_20d": row.rolling_volatility_20d,
        "volatility_regime_60d": row.volatility_regime_60d,
        "volume_zscore_20d": row.volume_zscore_20d,
        "volume_change_5d": row.volume_change_5d,
    }
    return data

