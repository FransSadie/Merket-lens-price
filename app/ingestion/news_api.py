import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class NewsApiClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.news_api_base_url.rstrip("/")

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def fetch_latest(self) -> List[Dict[str, Any]]:
        if not self.settings.news_api_key:
            raise ValueError("NEWS_API_KEY is missing. Set it in your .env file.")

        url = f"{self.base_url}/everything"
        page_size = max(1, min(self.settings.news_page_size, 100))
        max_pages = max(1, self.settings.news_max_pages)
        lookback_days = max(1, self.settings.news_lookback_days)
        from_dt = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).replace(microsecond=0)

        all_articles: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()

        fetched_pages = 0
        for page in range(1, max_pages + 1):
            params = {
                "q": self.settings.news_query,
                "language": self.settings.news_language,
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "page": page,
                "from": from_dt.isoformat().replace("+00:00", "Z"),
                "apiKey": self.settings.news_api_key,
            }
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()

            if payload.get("status") != "ok":
                raise RuntimeError(f"News API error: {payload}")

            page_articles = payload.get("articles", [])
            if not page_articles:
                break
            fetched_pages = page
            for article in page_articles:
                url_value = article.get("url")
                if not url_value or url_value in seen_urls:
                    continue
                seen_urls.add(url_value)
                all_articles.append(article)

            if len(page_articles) < page_size:
                break

        logger.info("Fetched %s unique articles from News API (%s pages)", len(all_articles), fetched_pages)
        return all_articles


def parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
