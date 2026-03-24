from datetime import UTC, datetime

from sqlalchemy import select

from app.core.logging import setup_logging
from app.db.models import NewsArticle
from app.db.schema import ensure_schema
from app.db.session import get_db_session


def seed_nlp_markers(batch_size: int = 5000) -> dict[str, int]:
    updated = 0
    last_id = 0
    processed_at = datetime.now(UTC).replace(tzinfo=None)
    session = get_db_session()
    try:
        while True:
            articles = session.execute(
                select(NewsArticle)
                .where(
                    NewsArticle.id > last_id,
                    NewsArticle.source_hash.is_not(None),
                    NewsArticle.nlp_processed_at.is_(None),
                )
                .order_by(NewsArticle.id.asc())
                .limit(batch_size)
            ).scalars().all()
            if not articles:
                break

            for article in articles:
                last_id = article.id
                article.nlp_source_hash = article.source_hash
                article.nlp_processed_at = processed_at
                updated += 1

            session.commit()
    finally:
        session.close()

    return {"rows_updated": updated}


def main() -> None:
    setup_logging()
    ensure_schema()
    result = seed_nlp_markers()
    print(result)


if __name__ == "__main__":
    main()
