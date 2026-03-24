import hashlib


def build_article_source_hash(
    source_name: str | None,
    title: str | None,
    description: str | None,
    content: str | None,
    url: str | None,
) -> str:
    payload = "||".join(
        [
            (source_name or "").strip(),
            (title or "").strip(),
            (description or "").strip(),
            (content or "").strip(),
            (url or "").strip(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
