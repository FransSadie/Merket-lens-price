from app.core.logging import setup_logging
from app.db.schema import ensure_schema
from app.ingestion.historical_news import run_historical_news_import


def main() -> None:
    setup_logging()
    ensure_schema()
    result = run_historical_news_import()
    print(result)


if __name__ == "__main__":
    main()
