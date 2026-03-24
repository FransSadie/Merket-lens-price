from datetime import datetime, timedelta

from sqlalchemy import func, select, text

from app.db.models import FeatureSnapshot, IngestionRun, MarketLabel, MarketPrice, NewsArticle, NewsSignal
from app.db.session import get_db_session


def data_status_snapshot() -> dict:
    session = get_db_session()
    try:
        news_total = session.execute(select(func.count(NewsArticle.id))).scalar_one()
        market_total = session.execute(select(func.count(MarketPrice.id))).scalar_one()
        signal_total = session.execute(select(func.count(NewsSignal.id))).scalar_one()
        feature_total = session.execute(select(func.count(FeatureSnapshot.id))).scalar_one()
        label_total = session.execute(select(func.count(MarketLabel.id))).scalar_one()

        market_ranges = session.execute(
            select(
                MarketPrice.ticker,
                func.count(MarketPrice.id),
                func.min(MarketPrice.timestamp),
                func.max(MarketPrice.timestamp),
            )
            .where(MarketPrice.interval == "1d")
            .group_by(MarketPrice.ticker)
            .order_by(MarketPrice.ticker)
        ).all()

        return {
            "totals": {
                "news_articles": news_total,
                "market_prices": market_total,
                "news_signals": signal_total,
                "feature_snapshots": feature_total,
                "market_labels": label_total,
            },
            "market_ranges": [
                {
                    "ticker": ticker,
                    "rows": int(rows),
                    "min_timestamp": min_ts.isoformat() if min_ts else None,
                    "max_timestamp": max_ts.isoformat() if max_ts else None,
                }
                for ticker, rows, min_ts, max_ts in market_ranges
            ],
        }
    finally:
        session.close()


def run_data_quality_checks() -> dict:
    session = get_db_session()
    try:
        duplicate_news = session.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                    SELECT source_name, url, COUNT(*) c
                    FROM news_articles
                    GROUP BY source_name, url
                    HAVING COUNT(*) > 1
                ) t
                """
            )
        ).scalar_one()
        duplicate_market = session.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                    SELECT ticker, timestamp, COUNT(*) c
                    FROM market_prices
                    GROUP BY ticker, timestamp
                    HAVING COUNT(*) > 1
                ) t
                """
            )
        ).scalar_one()
        null_published = session.execute(
            select(func.count(NewsArticle.id)).where(NewsArticle.published_at.is_(None))
        ).scalar_one()
        total_news = session.execute(select(func.count(NewsArticle.id))).scalar_one()

        latest_run = session.execute(select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(1)).scalar_one_or_none()
        latest_market = session.execute(select(func.max(MarketPrice.timestamp))).scalar_one()
        latest_news = session.execute(select(func.max(NewsArticle.published_at))).scalar_one()

        now = datetime.utcnow()
        stale_market = bool(latest_market and latest_market < now - timedelta(days=5))
        stale_news = bool(latest_news and latest_news < now - timedelta(days=3))

        return {
            "duplicate_keys": {
                "news_source_url_duplicates": int(duplicate_news),
                "market_ticker_timestamp_duplicates": int(duplicate_market),
            },
            "null_checks": {
                "news_published_at_nulls": int(null_published),
                "news_published_at_null_ratio": float(null_published / total_news) if total_news else 0.0,
            },
            "freshness": {
                "latest_ingestion_run": latest_run.started_at.isoformat() if latest_run else None,
                "latest_market_timestamp": latest_market.isoformat() if latest_market else None,
                "latest_news_published_at": latest_news.isoformat() if latest_news else None,
                "market_stale": stale_market,
                "news_stale": stale_news,
            },
        }
    finally:
        session.close()

