from app.core.article_hash import build_article_source_hash
from app.core.logging import setup_logging
from app.db.models import NewsArticle
from app.db.schema import ensure_schema
from app.db.session import get_db_session
from sqlalchemy import select


def seed_article_hashes(batch_size: int = 5000) -> dict[str, int]:
    updated = 0
    last_id = 0
    session = get_db_session()
    try:
        while True:
            articles = session.execute(
                select(NewsArticle)
                .where(NewsArticle.id > last_id, NewsArticle.source_hash.is_(None))
                .order_by(NewsArticle.id.asc())
                .limit(batch_size)
            ).scalars().all()
            if not articles:
                break

            for article in articles:
                last_id = article.id
                article.source_hash = build_article_source_hash(
                    source_name=article.source_name,
                    title=article.title,
                    description=article.description,
                    content=article.content,
                    url=article.url,
                )
                updated += 1

            session.commit()
    finally:
        session.close()

    return {"rows_updated": updated}


def main() -> None:
    setup_logging()
    ensure_schema()
    result = seed_article_hashes()
    print(result)


if __name__ == "__main__":
    main()
