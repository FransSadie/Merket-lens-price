from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class NewsArticle(Base):
    __tablename__ = "news_articles"
    __table_args__ = (UniqueConstraint("source_name", "url", name="uq_news_source_url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    tickers: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, default="live_api", index=True)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    import_batch: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    metadata_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    nlp_source_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    nlp_processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class MarketPrice(Base):
    __tablename__ = "market_prices"
    __table_args__ = (UniqueConstraint("ticker", "timestamp", name="uq_market_ticker_timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String(16), nullable=False, default="1d")
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    rows_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class NewsSignal(Base):
    __tablename__ = "news_signals"
    __table_args__ = (UniqueConstraint("article_id", "ticker", name="uq_news_signal_article_ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("news_articles.id"), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    event_tags: Mapped[str | None] = mapped_column(String(256), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"
    __table_args__ = (UniqueConstraint("ticker", "window_end", "window_hours", name="uq_feature_window"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, index=True)
    window_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=24)
    news_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sentiment_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_sum: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_weight_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    weighted_news_count: Mapped[float | None] = mapped_column(Float, nullable=True)
    weighted_sentiment_sum: Mapped[float | None] = mapped_column(Float, nullable=True)
    positive_news_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    negative_news_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_diversity: Mapped[float | None] = mapped_column(Float, nullable=True)
    news_count_72h: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_intensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    news_count_change_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_momentum_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_1d: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_return_3d: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_return_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_return_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_return_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    ma_gap_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    ma_gap_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    ma_crossover_5_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    range_pct_1d: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_14_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_volatility_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_regime_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_zscore_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_change_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class MarketLabel(Base):
    __tablename__ = "market_labels"
    __table_args__ = (UniqueConstraint("ticker", "timestamp", "horizon_days", name="uq_market_label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    target_up: Mapped[int] = mapped_column(Integer, nullable=False)
    future_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    prediction: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    probability_up: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False, index=True)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    version_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    training_feature_set: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    target_return_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    walk_forward_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    walk_forward_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    signals_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    signals_hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    signals_avg_future_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_hold_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False, index=True)
