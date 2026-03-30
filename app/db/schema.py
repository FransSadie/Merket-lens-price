import logging

from sqlalchemy import text

from app.db.base import Base
from app.db.session import engine

logger = logging.getLogger(__name__)


def _add_column_if_missing(table_name: str, column_name: str, definition: str) -> None:
    statement = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text(statement))
            trans.commit()
            logger.info("Added column %s.%s", table_name, column_name)
        except Exception as exc:
            trans.rollback()
            message = str(exc).lower()
            if "duplicate column" in message or "already exists" in message:
                return
            raise


def ensure_schema() -> None:
    Base.metadata.create_all(bind=engine)

    _add_column_if_missing("news_articles", "source_type", "VARCHAR(64) DEFAULT 'live_api'")
    _add_column_if_missing("news_articles", "external_id", "VARCHAR(255)")
    _add_column_if_missing("news_articles", "import_batch", "VARCHAR(255)")
    _add_column_if_missing("news_articles", "metadata_text", "TEXT")
    _add_column_if_missing("news_articles", "source_hash", "VARCHAR(64)")
    _add_column_if_missing("news_articles", "nlp_source_hash", "VARCHAR(64)")
    _add_column_if_missing("news_articles", "nlp_processed_at", "TIMESTAMP")

    _add_column_if_missing("news_signals", "source_weight", "DOUBLE PRECISION DEFAULT 1.0")
    _add_column_if_missing("news_signals", "event_tags", "VARCHAR(256)")

    feature_columns = [
        ("source_weight_mean", "DOUBLE PRECISION"),
        ("max_relevance", "DOUBLE PRECISION"),
        ("weighted_news_count", "DOUBLE PRECISION"),
        ("weighted_sentiment_sum", "DOUBLE PRECISION"),
        ("positive_news_ratio", "DOUBLE PRECISION"),
        ("negative_news_ratio", "DOUBLE PRECISION"),
        ("source_diversity", "DOUBLE PRECISION"),
        ("news_count_72h", "DOUBLE PRECISION"),
        ("event_intensity", "DOUBLE PRECISION"),
        ("news_count_change_24h", "DOUBLE PRECISION"),
        ("sentiment_momentum_24h", "DOUBLE PRECISION"),
        ("price_return_2d", "DOUBLE PRECISION"),
        ("price_return_3d", "DOUBLE PRECISION"),
        ("price_return_5d", "DOUBLE PRECISION"),
        ("price_return_10d", "DOUBLE PRECISION"),
        ("price_return_15d", "DOUBLE PRECISION"),
        ("price_return_20d", "DOUBLE PRECISION"),
        ("ma_gap_5d", "DOUBLE PRECISION"),
        ("ma_gap_20d", "DOUBLE PRECISION"),
        ("ma_crossover_5_20", "DOUBLE PRECISION"),
        ("ema_gap_12d", "DOUBLE PRECISION"),
        ("ema_gap_26d", "DOUBLE PRECISION"),
        ("ema_crossover_12_26", "DOUBLE PRECISION"),
        ("range_pct_1d", "DOUBLE PRECISION"),
        ("atr_14_pct", "DOUBLE PRECISION"),
        ("rolling_volatility_20d", "DOUBLE PRECISION"),
        ("downside_volatility_20d", "DOUBLE PRECISION"),
        ("volatility_regime_60d", "DOUBLE PRECISION"),
        ("volume_zscore_20d", "DOUBLE PRECISION"),
        ("volume_change_5d", "DOUBLE PRECISION"),
        ("volume_ratio_20d", "DOUBLE PRECISION"),
        ("rsi_5", "DOUBLE PRECISION"),
        ("rsi_14", "DOUBLE PRECISION"),
        ("bollinger_z_20", "DOUBLE PRECISION"),
        ("breakout_20d", "DOUBLE PRECISION"),
        ("drawdown_20d", "DOUBLE PRECISION"),
        ("trend_slope_10d", "DOUBLE PRECISION"),
        ("trend_slope_20d", "DOUBLE PRECISION"),
        ("up_day_ratio_10d", "DOUBLE PRECISION"),
        ("rel_to_spy_return_5d", "DOUBLE PRECISION"),
        ("rel_to_spy_return_20d", "DOUBLE PRECISION"),
        ("rel_to_qqq_return_5d", "DOUBLE PRECISION"),
        ("rel_to_sector_return_5d", "DOUBLE PRECISION"),
        ("rel_to_sector_return_20d", "DOUBLE PRECISION"),
        ("market_beta_20d", "DOUBLE PRECISION"),
        ("market_corr_20d", "DOUBLE PRECISION"),
        ("sector_corr_20d", "DOUBLE PRECISION"),
        ("volume_vs_spy_ratio_20d", "DOUBLE PRECISION"),
    ]
    for column_name, definition in feature_columns:
        _add_column_if_missing("feature_snapshots", column_name, definition)
