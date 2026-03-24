from app.core.logging import setup_logging
from app.db.schema import ensure_schema
from app.ingestion.pipeline import run_all_ingestion


def main() -> None:
    setup_logging()
    ensure_schema()
    result = run_all_ingestion()
    print(result)


if __name__ == "__main__":
    main()
