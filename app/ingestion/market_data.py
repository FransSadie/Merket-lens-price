import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_latest_prices(tickers: List[str], period: str = "5d", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )
        if df is None or df.empty:
            logger.warning("No market data returned for %s", ticker)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "Timestamp"})
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Timestamp"})
        if "Timestamp" not in df.columns:
            logger.warning("Missing timestamp column for %s", ticker)
            continue

        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
        data[ticker] = df
        logger.info("Fetched %s price rows for %s", len(df), ticker)
    return data


def to_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None

