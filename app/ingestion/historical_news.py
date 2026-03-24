import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

from sqlalchemy import select

from app.core.article_hash import build_article_source_hash
from app.core.config import get_settings
from app.db.models import IngestionRun, NewsArticle
from app.db.session import get_db_session
from app.ingestion.news_api import parse_published_at

logger = logging.getLogger(__name__)


CSV_COLUMN_ALIASES = {
    "title": ("title", "headline"),
    "url": ("url", "link", "documentidentifier"),
    "published_at": ("published_at", "published", "publish_date", "date", "datetime"),
    "source_name": ("source_name", "source", "sourcecommonname"),
    "description": ("description", "summary"),
    "content": ("content", "body", "text"),
    "external_id": ("external_id", "id", "documentid", "gkgrecordid"),
}


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


def run_historical_news_import(
    directory: str | None = None, file_format: str | None = None, batch_limit: int | None = None
) -> dict[str, int]:
    settings = get_settings()
    source_dir = Path(directory or settings.historical_news_dir)
    fmt = (file_format or settings.historical_news_format).lower()
    limit = batch_limit if batch_limit is not None else settings.historical_news_batch_limit
    run = _start_run("historical_news_import")

    if not source_dir.exists():
        message = f"Historical news directory not found: {source_dir}"
        _finish_run(run.id, "failed", 0, message)
        return {"files_processed": 0, "rows_inserted": 0, "rows_skipped": 0}

    try:
        files = _discover_files(source_dir, fmt, limit)
        inserted = 0
        skipped = 0
        updated = 0

        session = get_db_session()
        try:
            for file_path in files:
                for record in _iter_records(file_path, fmt):
                    normalized = _normalize_record(record, file_path)
                    if not normalized:
                        skipped += 1
                        continue

                    existing_article = session.execute(
                        select(NewsArticle).where(
                            NewsArticle.source_name == normalized["source_name"],
                            NewsArticle.url == normalized["url"],
                        )
                    ).scalar_one_or_none()
                    if existing_article:
                        if existing_article.source_type == "historical_batch":
                            current_hash = build_article_source_hash(
                                source_name=normalized["source_name"],
                                title=normalized["title"],
                                description=normalized.get("description"),
                                content=normalized.get("content"),
                                url=normalized["url"],
                            )
                            existing_article.title = normalized["title"][:1024]
                            existing_article.description = normalized.get("description")
                            existing_article.content = normalized.get("content")
                            existing_article.published_at = normalized.get("published_at")
                            existing_article.external_id = normalized.get("external_id")
                            existing_article.import_batch = file_path.name
                            existing_article.metadata_text = normalized.get("metadata_text")
                            existing_article.source_hash = current_hash
                            updated += 1
                        else:
                            skipped += 1
                        continue

                    session.add(
                        NewsArticle(
                            source_name=normalized["source_name"],
                            author=None,
                            title=normalized["title"][:1024],
                            description=normalized.get("description"),
                            content=normalized.get("content"),
                            url=normalized["url"][:2048],
                            published_at=normalized.get("published_at"),
                            tickers=None,
                            source_type="historical_batch",
                            external_id=normalized.get("external_id"),
                            import_batch=file_path.name,
                            metadata_text=normalized.get("metadata_text"),
                            source_hash=build_article_source_hash(
                                source_name=normalized["source_name"],
                                title=normalized["title"],
                                description=normalized.get("description"),
                                content=normalized.get("content"),
                                url=normalized["url"],
                            ),
                        )
                    )
                    inserted += 1

            session.commit()
        finally:
            session.close()

        message = f"Processed {len(files)} files, inserted {inserted}, updated {updated}, skipped {skipped}"
        _finish_run(run.id, "success", inserted, message)
        logger.info(message)
        return {
            "files_processed": len(files),
            "rows_inserted": inserted,
            "rows_updated": updated,
            "rows_skipped": skipped,
        }
    except Exception as exc:
        _finish_run(run.id, "failed", 0, str(exc))
        logger.exception("Historical news import failed")
        raise


def _discover_files(source_dir: Path, file_format: str, batch_limit: int) -> list[Path]:
    pattern = "*.jsonl" if file_format == "jsonl" else "*.csv"
    files = sorted(source_dir.glob(pattern))
    if batch_limit and batch_limit > 0:
        return files[:batch_limit]
    return files


def _iter_records(file_path: Path, file_format: str) -> Iterable[dict]:
    if file_format == "jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def _normalize_record(record: dict, file_path: Path) -> dict | None:
    lowered = {str(key).strip().lower(): value for key, value in record.items()}
    title = _get_alias_value(lowered, "title")
    url = _get_alias_value(lowered, "url")
    if not title or not url:
        return None

    published_raw = _get_alias_value(lowered, "published_at")
    source_name = _get_alias_value(lowered, "source_name") or "historical_import"
    description = _get_alias_value(lowered, "description")
    content = _get_alias_value(lowered, "content")
    external_id = _get_alias_value(lowered, "external_id")

    metadata = {
        "import_file": file_path.name,
        "raw_keys": sorted(lowered.keys()),
    }

    return {
        "title": str(title).strip(),
        "url": str(url).strip(),
        "published_at": _parse_historical_timestamp(published_raw),
        "source_name": str(source_name).strip(),
        "description": str(description).strip() if description else None,
        "content": str(content).strip() if content else None,
        "external_id": str(external_id).strip() if external_id else None,
        "metadata_text": json.dumps(metadata),
    }


def _get_alias_value(record: dict[str, object], canonical_name: str) -> object | None:
    for alias in CSV_COLUMN_ALIASES[canonical_name]:
        if alias in record and record[alias] not in (None, ""):
            return record[alias]
    return None


def _parse_historical_timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()

    if text.isdigit() and len(text) == 14:
        try:
            return datetime.strptime(text, "%Y%m%d%H%M%S")
        except ValueError:
            return None

    if text.isdigit() and len(text) == 8:
        try:
            return datetime.strptime(text, "%Y%m%d")
        except ValueError:
            return None

    return parse_published_at(text)
