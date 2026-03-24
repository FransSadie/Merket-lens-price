import logging
import re
import json
from datetime import datetime

from sqlalchemy import delete, or_, select

from app.core.article_hash import build_article_source_hash
from app.core.config import get_settings
from app.db.models import IngestionRun, NewsArticle, NewsSignal
from app.db.session import get_db_session

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
WORD_PATTERN = re.compile(r"[A-Za-z]+")

POSITIVE_WORDS = {
    "beat",
    "bullish",
    "growth",
    "gain",
    "gains",
    "positive",
    "rally",
    "record",
    "surge",
    "upside",
    "upgrade",
}

NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "cuts",
    "decline",
    "drop",
    "downgrade",
    "fall",
    "loss",
    "miss",
    "negative",
    "recession",
    "risk",
}

DEFAULT_TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple", "iphone", "ipad", "mac", "ios", "app store", "cupertino"),
    "MSFT": ("microsoft", "azure", "windows", "xbox", "github", "openai", "satya nadella"),
    "NVDA": ("nvidia", "geforce", "cuda", "gpu", "chipmaker", "semiconductor", "jensen huang"),
    "SPY": (
        "s&p",
        "s&p 500",
        "sp500",
        "wall street",
        "stocks",
        "stock market",
        "equities",
        "market",
        "economy",
        "inflation",
        "interest rate",
        "federal reserve",
        "fed",
        "recession",
        "treasury yields",
        "consumer confidence",
        "jobs report",
        "payrolls",
        "risk sentiment",
    ),
    "QQQ": (
        "nasdaq 100",
        "nasdaq",
        "tech stocks",
        "big tech",
        "technology",
        "semiconductor",
        "ai",
        "cloud",
        "software",
        "internet",
        "chip stocks",
    ),
}

TICKER_THEME_HINTS: dict[str, tuple[str, ...]] = {
    "AAPL": ("consumer electronics", "mobile", "smartphone"),
    "MSFT": ("cloud", "enterprise software", "artificial intelligence"),
    "NVDA": ("artificial intelligence", "semiconductor", "chips", "datacenter"),
    "SPY": ("economy", "inflation", "rates", "jobs", "gdp", "markets", "equities"),
    "QQQ": ("technology", "nasdaq", "software", "semiconductor", "ai", "internet"),
}

MARKET_PROXY_RULES: dict[str, tuple[str, ...]] = {
    "SPY": (
        "markets",
        "economy",
        "inflation",
        "rates",
        "federal reserve",
        "fed",
        "equities",
        "stocks",
        "s&p",
        "dow",
        "wall street",
        "gdp",
        "recession",
        "payrolls",
        "treasury",
    ),
    "QQQ": (
        "technology",
        "tech",
        "software",
        "internet",
        "nasdaq",
        "ai",
        "artificial intelligence",
        "cloud",
        "semiconductor",
        "chip",
    ),
}

ENTITY_TO_TICKER_HINTS: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple", "iphone", "ipad", "tim cook"),
    "MSFT": ("microsoft", "azure", "windows", "satya nadella", "xbox"),
    "NVDA": ("nvidia", "jensen huang", "gpu", "cuda", "chipmaker"),
}

SOURCE_WEIGHTS: dict[str, float] = {
    "reuters": 1.3,
    "bloomberg": 1.3,
    "wall street journal": 1.25,
    "financial times": 1.25,
    "cnbc": 1.15,
    "marketwatch": 1.1,
    "yahoo finance": 1.05,
}

EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": ("earnings", "eps", "guidance"),
    "rates": ("fed", "rate", "interest rate", "cut", "hike"),
    "inflation": ("inflation", "cpi", "ppi"),
    "m_and_a": ("acquire", "merger", "buyout"),
    "regulation": ("regulation", "antitrust", "lawsuit", "fine"),
}

NOISE_KEYWORDS = (
    "arrest",
    "murder",
    "shooting",
    "police",
    "crime",
    "trial",
    "weather",
    "storm",
    "snow",
    "rain",
    "earthquake",
    "sports",
    "football",
    "soccer",
    "baseball",
    "basketball",
    "airport",
    "celebrity",
    "entertainment",
    "music",
    "movie",
    "hollywood",
)

MARKET_CONTEXT_KEYWORDS = (
    "market",
    "markets",
    "stocks",
    "equities",
    "shares",
    "investor",
    "investors",
    "earnings",
    "guidance",
    "revenue",
    "profit",
    "fed",
    "rates",
    "inflation",
    "economy",
    "nasdaq",
    "s&p",
    "dow",
    "treasury",
    "yield",
)

MIN_RELEVANCE_BY_TICKER = {
    "AAPL": 0.05,
    "MSFT": 0.05,
    "NVDA": 0.05,
    "SPY": 0.08,
    "QQQ": 0.08,
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


def _sentiment_score(text: str) -> float:
    tokens = [token.lower() for token in WORD_PATTERN.findall(text)]
    if not tokens:
        return 0.0
    positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    score = (positive - negative) / max(len(tokens), 1)
    return max(min(score * 10, 1.0), -1.0)


def _extract_tickers(article: NewsArticle, universe: set[str], text: str) -> set[str]:
    tickers = set()
    structured = _parse_structured_content(article.content)
    structured_text = " ".join(
        [
            " ".join(structured.get("themes", [])),
            " ".join(structured.get("organizations", [])),
            " ".join(structured.get("persons", [])),
            " ".join(structured.get("locations", [])),
        ]
    ).strip()
    combined_text = f"{text} {structured_text}".strip()
    if article.tickers:
        tickers.update([t.strip().upper() for t in article.tickers.split(",") if t.strip()])
    tickers.update([m for m in TICKER_PATTERN.findall(combined_text.upper()) if m in universe])
    lower_text = combined_text.lower()
    for ticker in universe:
        if ticker.lower() in lower_text:
            tickers.add(ticker)
        aliases = _ticker_aliases(ticker)
        if any(alias in lower_text for alias in aliases):
            tickers.add(ticker)
        hints = TICKER_THEME_HINTS.get(ticker, ())
        if any(hint in lower_text for hint in hints):
            tickers.add(ticker)
        entity_hints = ENTITY_TO_TICKER_HINTS.get(ticker, ())
        if any(entity in lower_text for entity in entity_hints):
            tickers.add(ticker)

    tickers.update(_market_proxy_tickers(universe, lower_text, structured))
    return {ticker for ticker in tickers if ticker in universe}


def _relevance_score(text: str, ticker: str, article: NewsArticle) -> float:
    if not text and not article.content:
        return 0.0
    structured = _parse_structured_content(article.content)
    full_text = f"{text} {' '.join(structured.get('themes', []))} {' '.join(structured.get('organizations', []))}"
    lower_text = full_text.lower()
    aliases = _ticker_aliases(ticker)
    alias_hits = sum(lower_text.count(alias) for alias in aliases)
    symbol_hits = full_text.upper().count(ticker.upper())
    title_hits = sum((article.title or "").lower().count(alias) for alias in aliases)
    hint_hits = sum(lower_text.count(hint) for hint in TICKER_THEME_HINTS.get(ticker, ()))
    proxy_hits = sum(lower_text.count(keyword) for keyword in MARKET_PROXY_RULES.get(ticker, ()))
    tokens = max(len(WORD_PATTERN.findall(full_text)), 1)
    raw_score = ((symbol_hits * 2.0) + (alias_hits * 1.5) + (title_hits * 2.5) + hint_hits + (proxy_hits * 0.75)) / tokens
    return max(min(raw_score * 25, 1.0), 0.0)


def _is_article_noise(article: NewsArticle, text: str, structured: dict[str, list[str]]) -> bool:
    lower_text = text.lower()
    theme_text = " ".join(structured.get("themes", []))
    combined = f"{lower_text} {theme_text}"
    has_market_context = any(keyword in combined for keyword in MARKET_CONTEXT_KEYWORDS)
    noise_hits = sum(1 for keyword in NOISE_KEYWORDS if keyword in combined)

    if article.source_type != "historical_batch":
        return False

    if has_market_context:
        return False

    return noise_hits >= 1


def _should_keep_ticker_signal(article: NewsArticle, ticker: str, text: str, structured: dict[str, list[str]]) -> bool:
    if _is_article_noise(article, text, structured):
        return False

    relevance = _relevance_score(text, ticker, article)
    minimum = MIN_RELEVANCE_BY_TICKER.get(ticker, 0.03)
    if relevance < minimum:
        return False

    lower_text = text.lower()
    combined = f"{lower_text} {' '.join(structured.get('themes', []))} {' '.join(structured.get('organizations', []))}"
    if ticker in {"SPY", "QQQ"}:
        proxy_hits = sum(1 for keyword in MARKET_PROXY_RULES.get(ticker, ()) if keyword in combined)
        if proxy_hits < 2 and not any(alias in combined for alias in _ticker_aliases(ticker)):
            return False

    return True


def _source_weight(source_name: str) -> float:
    lowered = (source_name or "").lower()
    for key, weight in SOURCE_WEIGHTS.items():
        if key in lowered:
            return weight
    return 1.0


def _event_tags(text: str) -> str | None:
    lowered = (text or "").lower()
    tags = [tag for tag, keywords in EVENT_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]
    if any(keyword in lowered for keyword in ("market", "stocks", "equities", "shares", "nasdaq", "s&p")):
        tags.append("markets")
    if any(keyword in lowered for keyword in ("technology", "software", "semiconductor", "ai", "chip")):
        tags.append("technology")
    if any(keyword in lowered for keyword in ("gdp", "jobs", "payrolls", "consumer", "treasury", "economy")):
        tags.append("macro")
    if not tags:
        return None
    return ",".join(sorted(tags))


def _ticker_aliases(ticker: str) -> tuple[str, ...]:
    aliases = list(DEFAULT_TICKER_ALIASES.get(ticker, ()))
    aliases.append(ticker.lower())
    return tuple(dict.fromkeys(aliases))


def _parse_structured_content(content: str | None) -> dict[str, list[str]]:
    if not content:
        return {"themes": [], "organizations": [], "persons": [], "locations": []}
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {"themes": [], "organizations": [], "persons": [], "locations": []}

    return {
        "themes": _clean_list(payload.get("themes")),
        "organizations": _clean_list(payload.get("organizations")),
        "persons": _clean_list(payload.get("persons")),
        "locations": _clean_list(payload.get("locations")),
    }


def _clean_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        cleaned.append(text)
    return cleaned


def _market_proxy_tickers(
    universe: set[str], lower_text: str, structured: dict[str, list[str]]
) -> set[str]:
    proxy_tickers: set[str] = set()
    theme_text = " ".join(structured.get("themes", []))
    org_text = " ".join(structured.get("organizations", []))
    combined = f"{lower_text} {theme_text} {org_text}"

    for ticker, keywords in MARKET_PROXY_RULES.items():
        if ticker not in universe:
            continue
        hits = sum(1 for keyword in keywords if keyword in combined)
        if hits >= 2:
            proxy_tickers.add(ticker)

    if "SPY" in universe and "QQQ" in proxy_tickers and any(
        keyword in combined for keyword in ("market", "stocks", "equities", "economy", "fed", "inflation")
    ):
        proxy_tickers.add("SPY")

    return proxy_tickers


def run_news_nlp() -> int:
    settings = get_settings()
    run = _start_run("news_nlp")
    inserted = 0
    updated = 0
    universe = set(settings.market_ticker_list)
    try:
        session = get_db_session()
        try:
            last_id = 0
            batch_size = 1000
            while True:
                articles = session.execute(
                    select(NewsArticle)
                    .where(
                        NewsArticle.id > last_id,
                        or_(
                            NewsArticle.nlp_processed_at.is_(None),
                            NewsArticle.source_hash.is_(None),
                            NewsArticle.nlp_source_hash.is_(None),
                            NewsArticle.nlp_source_hash != NewsArticle.source_hash,
                        ),
                    )
                    .order_by(NewsArticle.id.asc())
                    .limit(batch_size)
                ).scalars().all()
                if not articles:
                    break

                for article in articles:
                    last_id = article.id
                    current_hash = article.source_hash or build_article_source_hash(
                        source_name=article.source_name,
                        title=article.title,
                        description=article.description,
                        content=article.content,
                        url=article.url,
                    )
                    article.source_hash = current_hash

                    text = f"{article.title or ''} {article.description or ''} {article.content or ''}".strip()
                    structured = _parse_structured_content(article.content)
                    candidate_tickers = _extract_tickers(article, universe, text)
                    desired_tickers = {
                        ticker
                        for ticker in candidate_tickers
                        if _should_keep_ticker_signal(article, ticker, text, structured)
                    }

                    existing_tickers = set(
                        session.execute(
                            select(NewsSignal.ticker).where(NewsSignal.article_id == article.id)
                        ).scalars().all()
                    )
                    stale_tickers = existing_tickers - desired_tickers
                    if stale_tickers:
                        session.execute(
                            delete(NewsSignal).where(
                                NewsSignal.article_id == article.id,
                                NewsSignal.ticker.in_(sorted(stale_tickers)),
                            )
                        )

                    if desired_tickers:
                        sentiment = _sentiment_score(text)
                        source_weight = _source_weight(article.source_name)
                        event_tags = _event_tags(text)
                        for ticker in desired_tickers:
                            relevance_score = _relevance_score(text, ticker, article)
                            existing_signal = session.execute(
                                select(NewsSignal).where(NewsSignal.article_id == article.id, NewsSignal.ticker == ticker)
                            ).scalar_one_or_none()
                            if existing_signal:
                                existing_signal.sentiment_score = max(min(sentiment * source_weight, 1.0), -1.0)
                                existing_signal.relevance_score = relevance_score
                                existing_signal.source_weight = source_weight
                                existing_signal.event_tags = event_tags
                                existing_signal.published_at = article.published_at
                                updated += 1
                                continue
                            signal = NewsSignal(
                                article_id=article.id,
                                ticker=ticker,
                                sentiment_score=max(min(sentiment * source_weight, 1.0), -1.0),
                                relevance_score=relevance_score,
                                source_weight=source_weight,
                                event_tags=event_tags,
                                published_at=article.published_at,
                            )
                            session.add(signal)
                            inserted += 1

                    article.nlp_source_hash = current_hash
                    article.nlp_processed_at = datetime.utcnow()

                session.commit()
        finally:
            session.close()
        _finish_run(run.id, "success", inserted, f"Inserted {inserted} news signals, updated {updated}")
        logger.info("News NLP complete: inserted=%s updated=%s", inserted, updated)
        return inserted
    except Exception as exc:
        _finish_run(run.id, "failed", inserted, str(exc))
        logger.exception("News NLP failed")
        return inserted
