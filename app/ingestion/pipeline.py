import logging
import re
from datetime import datetime
from typing import Iterable

from sqlalchemy import select

from app.core.article_hash import build_article_source_hash
from app.core.config import get_settings
from app.db.models import IngestionRun, MarketPrice, NewsArticle
from app.db.session import get_db_session
from app.ingestion.market_data import fetch_latest_prices, to_datetime
from app.ingestion.news_api import NewsApiClient, parse_published_at

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")


def _extract_tickers(text: str | None, universe: set[str]) -> str | None:
    if not text:
        return None
    matches = {m for m in TICKER_PATTERN.findall(text.upper()) if m in universe}
    if not matches:
        return None
    return ",".join(sorted(matches))


def _start_run(job_name: str) -> IngestionRun:
    session = get_db_session()
    try:
        run = IngestionRun(job_name=job_name, status="running", rows_inserted=0, started_at=datetime.utcnow())
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    finally:
        session.close()


def _finish_run(run_id: int, status: str, rows_inserted: int, message: str | None = None) -> None:
    session = get_db_session()
    try:
        run = session.get(IngestionRun, run_id)
        if not run:
            return
        run.status = status
        run.rows_inserted = rows_inserted
        run.message = message[:1000] if message else None
        run.completed_at = datetime.utcnow()
        session.commit()
    finally:
        session.close()


def run_news_ingestion() -> int:
    settings = get_settings()
    run = _start_run("news_ingestion")
    inserted = 0
    try:
        client = NewsApiClient()
        articles = client.fetch_latest()
        ticker_universe = set(settings.market_ticker_list)

        session = get_db_session()
        try:
            for article in articles:
                source_name = (article.get("source") or {}).get("name") or "unknown"
                url = article.get("url")
                title = article.get("title")
                if not url or not title:
                    continue

                existing = session.execute(
                    select(NewsArticle.id).where(NewsArticle.source_name == source_name, NewsArticle.url == url)
                ).scalar_one_or_none()
                if existing:
                    continue

                combined_text = f"{title} {article.get('description') or ''} {article.get('content') or ''}"
                model = NewsArticle(
                    source_name=source_name,
                    author=article.get("author"),
                    title=title[:1024],
                    description=article.get("description"),
                    content=article.get("content"),
                    url=url[:2048],
                    published_at=parse_published_at(article.get("publishedAt")),
                    tickers=_extract_tickers(combined_text, ticker_universe),
                    source_hash=build_article_source_hash(
                        source_name=source_name,
                        title=title,
                        description=article.get("description"),
                        content=article.get("content"),
                        url=url,
                    ),
                )
                session.add(model)
                inserted += 1

            session.commit()
        finally:
            session.close()
        _finish_run(run.id, "success", inserted, f"Inserted {inserted} news articles")
        logger.info("News ingestion complete: inserted=%s", inserted)
        return inserted
    except Exception as exc:
        _finish_run(run.id, "failed", inserted, str(exc))
        logger.exception("News ingestion failed")
        return inserted


def _iter_price_rows(data: dict) -> Iterable[tuple[str, dict]]:
    for ticker, frame in data.items():
        for _, row in frame.iterrows():
            yield ticker, row.to_dict()


def run_market_ingestion() -> int:
    settings = get_settings()
    run = _start_run("market_ingestion")
    inserted = 0
    try:
        if settings.market_history_interval != "1d":
            logger.warning(
                "Market ingestion interval is %s, but feature/label pipeline currently uses 1d rows.",
                settings.market_history_interval,
            )
        frames = fetch_latest_prices(
            settings.market_ticker_list,
            period=settings.market_history_period,
            interval=settings.market_history_interval,
        )

        session = get_db_session()
        try:
            for ticker, row in _iter_price_rows(frames):
                timestamp = to_datetime(row.get("Timestamp"))
                if not timestamp:
                    continue

                exists = session.execute(
                    select(MarketPrice.id).where(MarketPrice.ticker == ticker, MarketPrice.timestamp == timestamp)
                ).scalar_one_or_none()
                if exists:
                    continue

                model = MarketPrice(
                    ticker=ticker,
                    timestamp=timestamp,
                    interval=settings.market_history_interval,
                    open=_to_float(row.get("Open")),
                    high=_to_float(row.get("High")),
                    low=_to_float(row.get("Low")),
                    close=_to_float(row.get("Close")),
                    volume=_to_float(row.get("Volume")),
                )
                session.add(model)
                inserted += 1
            session.commit()
        finally:
            session.close()

        _finish_run(run.id, "success", inserted, f"Inserted {inserted} market rows")
        logger.info("Market ingestion complete: inserted=%s", inserted)
        return inserted
    except Exception as exc:
        _finish_run(run.id, "failed", inserted, str(exc))
        logger.exception("Market ingestion failed")
        return inserted


def run_active_ingestion() -> dict[str, int | bool]:
    settings = get_settings()
    news_count = run_news_ingestion() if settings.enable_news_pipeline else 0
    market_count = run_market_ingestion()
    return {
        "news_inserted": news_count,
        "market_inserted": market_count,
        "news_enabled": settings.enable_news_pipeline,
    }


def run_all_ingestion() -> dict[str, int | bool]:
    return run_active_ingestion()


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
