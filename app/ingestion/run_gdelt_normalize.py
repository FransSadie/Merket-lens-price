from app.core.logging import setup_logging
from app.ingestion.gdelt_gkg_normalizer import normalize_gdelt_gkg_batch


def main() -> None:
    setup_logging()
    result = normalize_gdelt_gkg_batch()
    print(result)


if __name__ == "__main__":
    main()
