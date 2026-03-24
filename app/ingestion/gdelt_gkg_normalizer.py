import csv
import io
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from app.core.config import get_settings

logger = logging.getLogger(__name__)


GKG_COLUMNS = (
    "gkgrecordid",
    "date",
    "source_collection_identifier",
    "sourcecommonname",
    "documentidentifier",
    "counts",
    "v2counts",
    "themes",
    "v2themes",
    "locations",
    "v2locations",
    "persons",
    "v2persons",
    "organizations",
    "v2organizations",
    "v2tone",
    "dates",
    "gcam",
    "sharingimage",
    "relatedimages",
    "socialimageembeds",
    "socialvideoembeds",
    "quotations",
    "allnames",
    "amounts",
    "translationinfo",
    "extrasxml",
)


@dataclass
class GkgNormalizationResult:
    files_processed: int
    rows_written: int
    rows_skipped: int
    output_files: list[str]


def normalize_gdelt_gkg_batch(
    input_dir: str | None = None,
    output_dir: str | None = None,
    batch_limit: int | None = None,
    row_limit: int | None = None,
) -> dict[str, object]:
    settings = get_settings()
    raw_dir = Path(input_dir or settings.gdelt_gkg_raw_dir)
    normalized_dir = Path(output_dir or settings.gdelt_gkg_normalized_dir)
    file_limit = batch_limit if batch_limit is not None else settings.gdelt_gkg_batch_limit
    max_rows = row_limit if row_limit is not None else settings.gdelt_gkg_row_limit

    normalized_dir.mkdir(parents=True, exist_ok=True)
    files = _discover_gkg_files(raw_dir, file_limit)

    if not raw_dir.exists():
        raise FileNotFoundError(f"GDELT raw directory not found: {raw_dir}")

    result = GkgNormalizationResult(files_processed=0, rows_written=0, rows_skipped=0, output_files=[])

    for file_path in files:
        output_path = normalized_dir / f"{file_path.stem}.normalized.csv"
        written, skipped = _normalize_single_file(file_path, output_path, max_rows=max_rows)
        result.files_processed += 1
        result.rows_written += written
        result.rows_skipped += skipped
        result.output_files.append(str(output_path))
        logger.info("Normalized %s -> %s (%s rows, %s skipped)", file_path.name, output_path.name, written, skipped)

    return {
        "files_processed": result.files_processed,
        "rows_written": result.rows_written,
        "rows_skipped": result.rows_skipped,
        "output_files": result.output_files,
    }


def _discover_gkg_files(raw_dir: Path, batch_limit: int) -> list[Path]:
    candidates = []
    for pattern in ("*.zip", "*.csv", "*.txt", "*.tsv"):
        candidates.extend(sorted(raw_dir.glob(pattern)))
    if batch_limit and batch_limit > 0:
        return candidates[:batch_limit]
    return candidates


def _normalize_single_file(file_path: Path, output_path: Path, max_rows: int) -> tuple[int, int]:
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "title",
                "url",
                "published_at",
                "source_name",
                "description",
                "content",
                "external_id",
            ),
        )
        writer.writeheader()

        for row in _iter_gkg_rows(file_path):
            if max_rows and max_rows > 0 and written >= max_rows:
                break

            normalized = _normalize_gkg_row(row)
            if not normalized:
                skipped += 1
                continue

            writer.writerow(normalized)
            written += 1

    return written, skipped


def _iter_gkg_rows(file_path: Path) -> Iterator[dict[str, str]]:
    if file_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(file_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if not members:
                return
            for member in members:
                with archive.open(member) as zipped_handle:
                    text_handle = io.TextIOWrapper(zipped_handle, encoding="utf-8", errors="replace")
                    yield from _read_gkg_text_stream(text_handle)
        return

    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from _read_gkg_text_stream(handle)


def _read_gkg_text_stream(handle: TextIO) -> Iterator[dict[str, str]]:
    first_line = handle.readline()
    if not first_line:
        return

    if _looks_like_header(first_line):
        reader = csv.DictReader(_prepend_line(handle, first_line))
        for row in reader:
            lowered = {str(key).strip().lower(): str(value).strip() for key, value in row.items() if key}
            yield lowered
        return

    yield _row_from_values(first_line.rstrip("\n").rstrip("\r").split("\t"))
    for line in handle:
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            continue
        yield _row_from_values(line.split("\t"))


def _prepend_line(handle: TextIO, first_line: str) -> Iterable[str]:
    yield first_line
    yield from handle


def _looks_like_header(first_line: str) -> bool:
    lowered = first_line.lower()
    return "documentidentifier" in lowered and "sourcecommonname" in lowered


def _row_from_values(values: list[str]) -> dict[str, str]:
    row: dict[str, str] = {}
    for index, column in enumerate(GKG_COLUMNS):
        row[column] = values[index].strip() if index < len(values) else ""
    return row


def _normalize_gkg_row(row: dict[str, str]) -> dict[str, str] | None:
    url = row.get("documentidentifier", "").strip()
    external_id = row.get("gkgrecordid", "").strip()
    if not url:
        return None

    source_name = row.get("sourcecommonname", "").strip() or "gdelt_gkg"
    published_at = row.get("date", "").strip()
    title = _build_title(row, source_name)
    description = _build_description(row)
    content = _build_content(row)

    if not title:
        return None

    return {
        "title": title,
        "url": url,
        "published_at": published_at,
        "source_name": source_name,
        "description": description,
        "content": content,
        "external_id": external_id,
    }


def _build_title(row: dict[str, str], source_name: str) -> str:
    themes = _extract_tokens(row.get("v2themes") or row.get("themes"))
    orgs = _extract_tokens(row.get("v2organizations") or row.get("organizations"))
    persons = _extract_tokens(row.get("persons"))
    headline_bits = []

    if themes:
        headline_bits.append(themes[0])
    if orgs:
        headline_bits.append(f"organizations {', '.join(orgs[:2])}")
    if persons:
        headline_bits.append(f"people {', '.join(persons[:2])}")

    if headline_bits:
        return f"{source_name}: {' | '.join(headline_bits)}"
    return f"{source_name}: GDELT historical article"


def _build_description(row: dict[str, str]) -> str | None:
    themes = _extract_theme_tokens(row.get("v2themes") or row.get("themes"), limit=5)
    orgs = _extract_tokens(row.get("v2organizations") or row.get("organizations"), limit=4)
    persons = _extract_tokens(row.get("persons"), limit=4)
    tone = _extract_tone(row.get("v2tone"))

    parts: list[str] = []
    if themes:
        parts.append(f"themes={', '.join(themes)}")
    if orgs:
        parts.append(f"organizations={', '.join(orgs)}")
    if persons:
        parts.append(f"persons={', '.join(persons)}")
    if tone is not None:
        parts.append(f"tone={tone}")

    return " | ".join(parts) if parts else None


def _build_content(row: dict[str, str]) -> str | None:
    themes = _extract_theme_tokens(row.get("v2themes") or row.get("themes"), limit=10)
    orgs = _extract_tokens(row.get("v2organizations") or row.get("organizations"), limit=10)
    persons = _extract_tokens(row.get("v2persons") or row.get("persons"), limit=10)
    payload = {
        "themes": themes,
        "organizations": orgs,
        "persons": persons,
        "locations": _extract_tokens(row.get("locations"), limit=6),
        "tone": _extract_tone(row.get("v2tone")),
        "counts": _extract_tokens(row.get("v2counts") or row.get("counts"), limit=8),
        "translationinfo": row.get("translationinfo", "").strip() or None,
    }
    compact = {key: value for key, value in payload.items() if value not in (None, [], "")}
    if not compact:
        return None
    return json.dumps(compact, ensure_ascii=True)


def _extract_tokens(value: str | None, limit: int = 3) -> list[str]:
    if not value:
        return []

    tokens: list[str] = []
    for raw in str(value).split(";"):
        cleaned = raw.strip()
        if not cleaned:
            continue
        if "," in cleaned:
            cleaned = cleaned.split(",")[0].strip()
        if "#" in cleaned:
            cleaned = cleaned.split("#")[0].strip()
        cleaned = cleaned.replace("_", " ").strip()
        if cleaned and cleaned not in tokens:
            tokens.append(cleaned)
        if len(tokens) >= limit:
            break
    return tokens


def _extract_theme_tokens(value: str | None, limit: int = 3) -> list[str]:
    raw_tokens = _extract_tokens(value, limit=limit * 3)
    cleaned: list[str] = []
    noise_prefixes = ("tax ", "wb ", "crisislex ", "ungp ", "epu ")
    for token in raw_tokens:
        candidate = token.lower().strip()
        for prefix in noise_prefixes:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix):].strip()
                break
        if candidate:
            candidate = candidate.replace("  ", " ")
        if candidate and candidate not in cleaned:
            cleaned.append(candidate)
        if len(cleaned) >= limit:
            break
    return cleaned


def _extract_tone(value: str | None) -> float | None:
    if not value:
        return None
    first = str(value).split(",")[0].strip()
    try:
        return float(first)
    except ValueError:
        return None
