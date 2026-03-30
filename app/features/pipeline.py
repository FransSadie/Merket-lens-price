import logging
from datetime import datetime, timedelta

from sqlalchemy import select

from app.db.models import FeatureSnapshot, IngestionRun, MarketLabel, MarketPrice, NewsArticle, NewsSignal
from app.db.session import get_db_session

logger = logging.getLogger(__name__)


SECTOR_BENCHMARKS = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AMZN": "XLY",
    "GOOGL": "XLK",
    "META": "XLK",
    "TSLA": "XLY",
    "AMD": "XLK",
    "AVGO": "XLK",
    "CRM": "XLK",
    "ORCL": "XLK",
    "ADBE": "XLK",
    "NFLX": "XLY",
    "CSCO": "XLK",
    "QCOM": "XLK",
    "JPM": "XLF",
    "GS": "XLF",
    "MS": "XLF",
    "V": "XLF",
    "MA": "XLF",
    "UNH": "XLV",
    "LLY": "XLV",
    "COST": "XLY",
    "WMT": "XLY",
    "KO": "XLP",
    "PEP": "XLP",
    "XOM": "XLE",
    "CVX": "XLE",
    "CAT": "XLI",
    "BA": "XLI",
    "GE": "XLI",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "DIA": "DIA",
    "XLF": "XLF",
    "XLK": "XLK",
    "XLV": "XLV",
    "XLY": "XLY",
    "XLI": "XLI",
    "XLE": "XLE",
    "XLB": "XLB",
    "XLU": "XLU",
    "XLP": "XLP",
    "TLT": "TLT",
    "GLD": "GLD",
}


PRICE_BACKFILL_FIELDS = [
    "price_return_2d",
    "price_return_3d",
    "price_return_5d",
    "price_return_10d",
    "price_return_15d",
    "price_return_20d",
    "ma_gap_5d",
    "ma_gap_20d",
    "ma_crossover_5_20",
    "ema_gap_12d",
    "ema_gap_26d",
    "ema_crossover_12_26",
    "range_pct_1d",
    "atr_14_pct",
    "rolling_volatility_20d",
    "downside_volatility_20d",
    "volatility_regime_60d",
    "volume_zscore_20d",
    "volume_change_5d",
    "volume_ratio_20d",
    "rsi_5",
    "rsi_14",
    "bollinger_z_20",
    "breakout_20d",
    "drawdown_20d",
    "trend_slope_10d",
    "trend_slope_20d",
    "up_day_ratio_10d",
    "rel_to_spy_return_5d",
    "rel_to_spy_return_20d",
    "rel_to_qqq_return_5d",
    "rel_to_sector_return_5d",
    "rel_to_sector_return_20d",
    "market_beta_20d",
    "market_corr_20d",
    "sector_corr_20d",
    "volume_vs_spy_ratio_20d",
]

logger = logging.getLogger(__name__)


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


def run_feature_generation(window_hours: int = 24) -> int:
    run = _start_run("feature_generation")
    inserted = 0
    updated = 0
    try:
        session = get_db_session()
        try:
            prices = session.execute(select(MarketPrice).where(MarketPrice.interval == "1d")).scalars().all()
            price_index = _build_price_index(prices)
            timestamp_lookup = _build_timestamp_lookup(price_index)
            existing_rows = session.execute(
                select(FeatureSnapshot).where(FeatureSnapshot.window_hours == window_hours)
            ).scalars().all()
            existing_map = {
                (row.ticker, row.window_end, row.window_hours): row
                for row in existing_rows
            }
            latest_rows = session.execute(
                select(FeatureSnapshot)
                .where(FeatureSnapshot.window_hours == window_hours)
                .order_by(FeatureSnapshot.ticker.asc(), FeatureSnapshot.window_end.desc())
            ).scalars().all()
            target_rows = _feature_target_rows(session, price_index, run.started_at, window_hours)
            if not target_rows:
                session.commit()
                _finish_run(run.id, "success", 0, "Inserted 0 feature rows, updated 0")
                logger.info("Feature generation complete: inserted=0 updated=0")
                return 0

            min_window_start = min(target.window_end for target in target_rows) - timedelta(hours=window_hours)
            max_window_end = max(target.window_end for target in target_rows)
            signals = session.execute(
                select(NewsSignal)
                .where(
                    NewsSignal.published_at.is_not(None),
                    NewsSignal.published_at >= min_window_start,
                    NewsSignal.published_at <= max_window_end,
                )
                .order_by(NewsSignal.ticker, NewsSignal.published_at)
            ).scalars().all()
            signals_by_ticker = _group_signals(signals)
            computed_states: dict[tuple[str, datetime], dict[str, float | None]] = {}

            for target in target_rows:
                ticker = target.ticker
                rows = price_index[ticker]
                idx = target.price_index
                price_row = rows[idx]
                window_end = price_row.timestamp
                feature_key = (ticker, window_end, window_hours)
                existing_row = existing_map.get(feature_key)
                ticker_signals = signals_by_ticker.get(ticker, [])
                window_start = window_end - timedelta(hours=window_hours)
                in_window = _signals_in_window(ticker_signals, window_start, window_end)
                if existing_row and not _should_refresh_feature_row(existing_row, in_window):
                    computed_states[(ticker, window_end)] = {
                        "news_count": float(existing_row.news_count),
                        "sentiment_mean": existing_row.sentiment_mean,
                    }
                    continue

                sentiments = [item.sentiment_score for item in in_window]
                relevances = [item.relevance_score for item in in_window]
                weights = [item.source_weight for item in in_window]
                weighted_sentiments = [item.sentiment_score * max(item.relevance_score, 0.05) * item.source_weight for item in in_window]
                event_intensity = _event_intensity(in_window)
                positive_news_ratio = _ratio([value for value in sentiments if value > 0], sentiments)
                negative_news_ratio = _ratio([value for value in sentiments if value < 0], sentiments)
                source_diversity = _source_diversity(in_window)
                news_count_72h = float(_count_signals_in_window(ticker_signals, window_end - timedelta(hours=72), window_end))

                price_features = _price_features(ticker, price_index, timestamp_lookup, idx)
                close_price = price_features["price_close"]
                return_1d = price_features["return_1d"]
                news_count = len(in_window)
                sentiment_mean = _mean(sentiments)
                previous = _previous_feature_state(ticker, rows, idx, existing_map, computed_states)
                news_count_change = (
                    float(news_count - previous["news_count"]) if previous["news_count"] is not None else None
                )
                sentiment_momentum = (
                    sentiment_mean - previous["sentiment_mean"]
                    if sentiment_mean is not None and previous["sentiment_mean"] is not None
                    else None
                )

                if existing_row:
                    row = existing_row
                    updated += 1
                else:
                    row = FeatureSnapshot(ticker=ticker, window_end=window_end, window_hours=window_hours)
                    session.add(row)
                    inserted += 1
                row.news_count = news_count
                row.sentiment_mean = sentiment_mean
                row.sentiment_sum = sum(sentiments) if sentiments else 0.0
                row.sentiment_std = _std(sentiments)
                row.relevance_mean = _mean(relevances)
                row.max_relevance = max(relevances) if relevances else None
                row.source_weight_mean = _mean(weights)
                row.weighted_news_count = sum(max(item.relevance_score, 0.05) * item.source_weight for item in in_window) if in_window else 0.0
                row.weighted_sentiment_sum = sum(weighted_sentiments) if weighted_sentiments else 0.0
                row.positive_news_ratio = positive_news_ratio
                row.negative_news_ratio = negative_news_ratio
                row.source_diversity = source_diversity
                row.news_count_72h = news_count_72h
                row.event_intensity = event_intensity
                row.news_count_change_24h = news_count_change
                row.sentiment_momentum_24h = sentiment_momentum
                row.price_close = close_price
                row.return_1d = return_1d
                for field_name, value in price_features.items():
                    setattr(row, field_name, value)
                computed_states[(ticker, window_end)] = {"news_count": float(news_count), "sentiment_mean": sentiment_mean}
            session.commit()
        finally:
            session.close()
        _finish_run(run.id, "success", inserted, f"Inserted {inserted} feature rows, updated {updated}")
        logger.info("Feature generation complete: inserted=%s updated=%s", inserted, updated)
        return inserted
    except Exception as exc:
        _finish_run(run.id, "failed", inserted, str(exc))
        logger.exception("Feature generation failed")
        return inserted


def run_label_generation(horizon_days: int = 1) -> int:
    run = _start_run("label_generation")
    inserted = 0
    try:
        session = get_db_session()
        try:
            prices = session.execute(select(MarketPrice).where(MarketPrice.interval == "1d")).scalars().all()
            by_ticker = _build_price_index(prices)
            for ticker, rows in by_ticker.items():
                for index in range(len(rows) - horizon_days):
                    current = rows[index]
                    future = rows[index + horizon_days]
                    if current.close is None or future.close is None or current.close == 0:
                        continue

                    exists = session.execute(
                        select(MarketLabel.id).where(
                            MarketLabel.ticker == ticker,
                            MarketLabel.timestamp == current.timestamp,
                            MarketLabel.horizon_days == horizon_days,
                        )
                    ).scalar_one_or_none()
                    if exists:
                        continue

                    future_return = (future.close - current.close) / current.close
                    row = MarketLabel(
                        ticker=ticker,
                        timestamp=current.timestamp,
                        horizon_days=horizon_days,
                        target_up=1 if future_return > 0 else 0,
                        future_return=future_return,
                    )
                    session.add(row)
                    inserted += 1
            session.commit()
        finally:
            session.close()
        _finish_run(run.id, "success", inserted, f"Inserted {inserted} labels")
        logger.info("Label generation complete: inserted=%s", inserted)
        return inserted
    except Exception as exc:
        _finish_run(run.id, "failed", inserted, str(exc))
        logger.exception("Label generation failed")
        return inserted


def _group_signals(signals: list[NewsSignal]) -> dict[str, list[NewsSignal]]:
    grouped: dict[str, list[NewsSignal]] = {}
    for signal in signals:
        grouped.setdefault(signal.ticker, []).append(signal)
    return grouped


class _FeatureTarget:
    def __init__(self, ticker: str, price_index: int, window_end: datetime) -> None:
        self.ticker = ticker
        self.price_index = price_index
        self.window_end = window_end


def _feature_target_rows(
    session, price_index: dict[str, list[MarketPrice]], current_run_started_at: datetime, window_hours: int
) -> list[_FeatureTarget]:
    since_time = _last_successful_run_completed_at(session, "feature_generation", current_run_started_at)
    if since_time is None:
        return _all_feature_targets(price_index)

    processed_articles = session.execute(
        select(NewsArticle.published_at)
        .where(
            NewsArticle.published_at.is_not(None),
            NewsArticle.nlp_processed_at.is_not(None),
            NewsArticle.nlp_processed_at > since_time,
        )
        .order_by(NewsArticle.published_at.asc())
    ).scalars().all()

    target_keys: set[tuple[str, datetime]] = set()
    if processed_articles:
        intervals = [(published_at, published_at + timedelta(hours=window_hours)) for published_at in processed_articles]
        for ticker, rows in price_index.items():
            for price_row in rows:
                if any(start <= price_row.timestamp <= end for start, end in intervals):
                    target_keys.add((ticker, price_row.timestamp))

    stale_rows = session.execute(
        select(FeatureSnapshot)
        .where(FeatureSnapshot.window_hours == window_hours)
    ).scalars().all()
    existing_feature_keys = {(row.ticker, row.window_end) for row in stale_rows}
    for row in stale_rows:
        if _needs_feature_backfill(row):
            target_keys.add((row.ticker, row.window_end))

    for ticker, rows in price_index.items():
        for price_row in rows:
            if (ticker, price_row.timestamp) not in existing_feature_keys:
                target_keys.add((ticker, price_row.timestamp))

    if not target_keys:
        return []

    targets: list[_FeatureTarget] = []
    for ticker, rows in price_index.items():
        for idx, price_row in enumerate(rows):
            if (ticker, price_row.timestamp) in target_keys:
                targets.append(_FeatureTarget(ticker=ticker, price_index=idx, window_end=price_row.timestamp))
    targets.sort(key=lambda item: (item.ticker, item.window_end))
    return targets


def _all_feature_targets(price_index: dict[str, list[MarketPrice]]) -> list[_FeatureTarget]:
    targets: list[_FeatureTarget] = []
    for ticker, rows in price_index.items():
        for idx, row in enumerate(rows):
            targets.append(_FeatureTarget(ticker=ticker, price_index=idx, window_end=row.timestamp))
    targets.sort(key=lambda item: (item.ticker, item.window_end))
    return targets


def _last_successful_run_completed_at(session, job_name: str, before_started_at: datetime) -> datetime | None:
    row = session.execute(
        select(IngestionRun)
        .where(
            IngestionRun.job_name == job_name,
            IngestionRun.status == "success",
            IngestionRun.completed_at.is_not(None),
            IngestionRun.started_at < before_started_at,
        )
        .order_by(IngestionRun.started_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if not row:
        return None
    return row.completed_at


def _signals_in_window(signals: list[NewsSignal], start: datetime, end: datetime) -> list[NewsSignal]:
    return [signal for signal in signals if signal.published_at and start <= signal.published_at <= end]


def _event_intensity(signals: list[NewsSignal]) -> float | None:
    if not signals:
        return None
    tagged = sum(1 for signal in signals if signal.event_tags)
    return tagged / len(signals)


def _count_signals_in_window(signals: list[NewsSignal], start: datetime, end: datetime) -> int:
    return sum(1 for signal in signals if signal.published_at and start <= signal.published_at <= end)


def _source_diversity(signals: list[NewsSignal]) -> float | None:
    if not signals:
        return None
    weights = [signal.source_weight for signal in signals if signal.source_weight is not None]
    if not weights:
        return None
    return len({round(weight, 3) for weight in weights}) / len(weights)


def _build_price_index(prices: list[MarketPrice]) -> dict[str, list[MarketPrice]]:
    index: dict[str, list[MarketPrice]] = {}
    for row in prices:
        index.setdefault(row.ticker, []).append(row)
    for ticker, rows in index.items():
        rows.sort(key=lambda x: x.timestamp)
        index[ticker] = rows
    return index


def _build_timestamp_lookup(price_index: dict[str, list[MarketPrice]]) -> dict[str, dict[datetime, MarketPrice]]:
    return {ticker: {row.timestamp: row for row in rows} for ticker, rows in price_index.items()}


def _price_return(rows: list[MarketPrice], current_index: int, lookback: int) -> float | None:
    if current_index < lookback:
        return None
    current_close = rows[current_index].close
    prev_close = rows[current_index - lookback].close
    if current_close is None or prev_close in (None, 0):
        return None
    return (current_close - prev_close) / prev_close


def _window_mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _window_std(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    avg = sum(clean) / len(clean)
    variance = sum((value - avg) ** 2 for value in clean) / len(clean)
    return variance ** 0.5


def _ema(values: list[float | None], period: int) -> float | None:
    clean = [value for value in values if value is not None]
    if len(clean) < period:
        return None
    alpha = 2 / (period + 1)
    ema_value = clean[0]
    for value in clean[1:]:
        ema_value = alpha * value + (1 - alpha) * ema_value
    return ema_value


def _rsi(rows: list[MarketPrice], current_index: int, period: int) -> float | None:
    if current_index < period:
        return None
    gains = []
    losses = []
    for idx in range(current_index - period + 1, current_index + 1):
        prev_close = rows[idx - 1].close
        curr_close = rows[idx].close
        if prev_close is None or curr_close is None:
            continue
        delta = curr_close - prev_close
        gains.append(max(delta, 0.0))
        losses.append(abs(min(delta, 0.0)))
    if not gains or not losses:
        return None
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _linear_slope(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return None
    n = len(clean)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(clean) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, clean))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = numerator / denominator
    if y_mean == 0:
        return slope
    return slope / y_mean


def _return_series(rows: list[MarketPrice], current_index: int, period: int) -> list[float]:
    start = max(1, current_index - period + 1)
    series = []
    for idx in range(start, current_index + 1):
        prev_close = rows[idx - 1].close
        curr_close = rows[idx].close
        if prev_close is not None and prev_close != 0 and curr_close is not None:
            series.append((curr_close - prev_close) / prev_close)
    return series


def _aligned_return_series(base_rows: list[MarketPrice], base_index: int, ref_lookup: dict[datetime, MarketPrice], period: int) -> tuple[list[float], list[float]]:
    base_returns = []
    ref_returns = []
    start = max(1, base_index - period + 1)
    for idx in range(start, base_index + 1):
        base_prev = base_rows[idx - 1]
        base_curr = base_rows[idx]
        ref_prev = ref_lookup.get(base_prev.timestamp)
        ref_curr = ref_lookup.get(base_curr.timestamp)
        if not ref_prev or not ref_curr:
            continue
        if base_prev.close in (None, 0) or base_curr.close is None:
            continue
        if ref_prev.close in (None, 0) or ref_curr.close is None:
            continue
        base_returns.append((base_curr.close - base_prev.close) / base_prev.close)
        ref_returns.append((ref_curr.close - ref_prev.close) / ref_prev.close)
    return base_returns, ref_returns


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / ((x_var ** 0.5) * (y_var ** 0.5))


def _beta(xs: list[float], market: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(market):
        return None
    market_mean = sum(market) / len(market)
    var_market = sum((m - market_mean) ** 2 for m in market) / len(market)
    if var_market == 0:
        return None
    asset_mean = sum(xs) / len(xs)
    cov = sum((x - asset_mean) * (m - market_mean) for x, m in zip(xs, market)) / len(xs)
    return cov / var_market


def _reference_return(ref_lookup: dict[datetime, MarketPrice], current_ts: datetime, lookback_ts: datetime | None) -> float | None:
    if lookback_ts is None:
        return None
    current_row = ref_lookup.get(current_ts)
    prev_row = ref_lookup.get(lookback_ts)
    if not current_row or not prev_row:
        return None
    if current_row.close is None or prev_row.close in (None, 0):
        return None
    return (current_row.close - prev_row.close) / prev_row.close


def _volume_ratio_against_reference(current_volume: float | None, ref_lookup: dict[datetime, MarketPrice], timestamps: list[datetime]) -> float | None:
    if current_volume is None:
        return None
    ref_volumes = []
    for ts in timestamps:
        row = ref_lookup.get(ts)
        if row and row.volume is not None:
            ref_volumes.append(row.volume)
    avg_ref_volume = _window_mean(ref_volumes)
    if avg_ref_volume in (None, 0):
        return None
    return current_volume / avg_ref_volume


def _price_features(ticker: str, price_index: dict[str, list[MarketPrice]], timestamp_lookup: dict[str, dict[datetime, MarketPrice]], current_index: int) -> dict[str, float | None]:
    price_rows = price_index[ticker]
    feature_keys = [
        "price_close", "return_1d", "price_return_2d", "price_return_3d", "price_return_5d", "price_return_10d",
        "price_return_15d", "price_return_20d", "ma_gap_5d", "ma_gap_20d", "ma_crossover_5_20", "ema_gap_12d",
        "ema_gap_26d", "ema_crossover_12_26", "range_pct_1d", "atr_14_pct", "rolling_volatility_20d",
        "downside_volatility_20d", "volatility_regime_60d", "volume_zscore_20d", "volume_change_5d", "volume_ratio_20d",
        "rsi_5", "rsi_14", "bollinger_z_20", "breakout_20d", "drawdown_20d", "trend_slope_10d", "trend_slope_20d",
        "up_day_ratio_10d", "rel_to_spy_return_5d", "rel_to_spy_return_20d", "rel_to_qqq_return_5d",
        "rel_to_sector_return_5d", "rel_to_sector_return_20d", "market_beta_20d", "market_corr_20d", "sector_corr_20d",
        "volume_vs_spy_ratio_20d",
    ]
    if not price_rows or current_index < 0 or current_index >= len(price_rows):
        return {field: None for field in feature_keys}

    current = price_rows[current_index]
    close_price = current.close

    next_return = None
    if close_price is not None and close_price != 0 and current_index + 1 < len(price_rows):
        next_close = price_rows[current_index + 1].close
        if next_close is not None:
            next_return = (next_close - close_price) / close_price

    close_window_5 = [row.close for row in price_rows[max(0, current_index - 4): current_index + 1]]
    close_window_10 = [row.close for row in price_rows[max(0, current_index - 9): current_index + 1]]
    close_window_20 = [row.close for row in price_rows[max(0, current_index - 19): current_index + 1]]
    close_window_26 = [row.close for row in price_rows[max(0, current_index - 25): current_index + 1]]

    ma_5 = _window_mean(close_window_5)
    ma_20 = _window_mean(close_window_20)
    ema_12 = _ema([row.close for row in price_rows[max(0, current_index - 30): current_index + 1]], 12)
    ema_26 = _ema(close_window_26, 26)

    ma_gap_5d = ((close_price - ma_5) / ma_5) if close_price is not None and ma_5 not in (None, 0) else None
    ma_gap_20d = ((close_price - ma_20) / ma_20) if close_price is not None and ma_20 not in (None, 0) else None
    ma_crossover_5_20 = ((ma_5 - ma_20) / ma_20) if ma_5 not in (None, 0) and ma_20 not in (None, 0) else None
    ema_gap_12d = ((close_price - ema_12) / ema_12) if close_price is not None and ema_12 not in (None, 0) else None
    ema_gap_26d = ((close_price - ema_26) / ema_26) if close_price is not None and ema_26 not in (None, 0) else None
    ema_crossover_12_26 = ((ema_12 - ema_26) / ema_26) if ema_12 not in (None, 0) and ema_26 not in (None, 0) else None

    range_pct_1d = None
    if current.high is not None and current.low is not None and close_price not in (None, 0):
        range_pct_1d = (current.high - current.low) / close_price

    true_ranges = []
    start_idx = max(1, current_index - 13)
    for i in range(start_idx, current_index + 1):
        row = price_rows[i]
        prev_close = price_rows[i - 1].close
        if row.high is None or row.low is None or prev_close is None:
            continue
        true_ranges.append(max(row.high - row.low, abs(row.high - prev_close), abs(row.low - prev_close)))
    atr_14 = _window_mean(true_ranges)
    atr_14_pct = (atr_14 / close_price) if atr_14 is not None and close_price not in (None, 0) else None

    returns_20 = _return_series(price_rows, current_index, 20)
    rolling_volatility_20d = _window_std(returns_20)
    downside_returns_20 = [value for value in returns_20 if value < 0]
    downside_volatility_20d = _window_std(downside_returns_20) if downside_returns_20 else 0.0

    vol_history = []
    if current_index >= 40:
        for end in range(current_index - 19, current_index + 1):
            vol_history.append(_window_std(_return_series(price_rows, end, 20)))
    clean_vols = [value for value in vol_history if value is not None]
    volatility_regime_60d = None
    if rolling_volatility_20d is not None and len(clean_vols) >= 5:
        avg_vol = _window_mean(clean_vols)
        sd_vol = _window_std(clean_vols)
        if avg_vol is not None and sd_vol not in (None, 0):
            volatility_regime_60d = (rolling_volatility_20d - avg_vol) / sd_vol

    volume_zscore_20d = None
    volume_ratio_20d = None
    volume_change_5d = None
    volumes_20 = [row.volume for row in price_rows[max(0, current_index - 19): current_index + 1]]
    clean_volumes = [v for v in volumes_20 if v is not None]
    if current.volume is not None and len(clean_volumes) >= 5:
        avg_volume = _window_mean(clean_volumes)
        sd_volume = _window_std(clean_volumes)
        if avg_volume not in (None, 0):
            volume_ratio_20d = current.volume / avg_volume
        if sd_volume not in (None, 0) and avg_volume is not None:
            volume_zscore_20d = (current.volume - avg_volume) / sd_volume
    if current.volume is not None and current_index >= 5:
        prev_volume = price_rows[current_index - 5].volume
        if prev_volume not in (None, 0):
            volume_change_5d = (current.volume - prev_volume) / prev_volume

    bollinger_z_20 = None
    if close_price is not None and ma_20 is not None:
        std20 = _window_std(close_window_20)
        if std20 not in (None, 0):
            bollinger_z_20 = (close_price - ma_20) / std20

    breakout_20d = None
    drawdown_20d = None
    if close_price is not None:
        highs_20 = [row.high for row in price_rows[max(0, current_index - 19): current_index + 1] if row.high is not None]
        closes_20 = [row.close for row in price_rows[max(0, current_index - 19): current_index + 1] if row.close is not None]
        if highs_20:
            breakout_20d = close_price / max(highs_20) - 1.0
        if closes_20:
            drawdown_20d = close_price / max(closes_20) - 1.0

    returns_10 = _return_series(price_rows, current_index, 10)
    up_day_ratio_10d = (len([value for value in returns_10 if value > 0]) / len(returns_10)) if returns_10 else None

    benchmark_spy = timestamp_lookup.get("SPY", {})
    benchmark_qqq = timestamp_lookup.get("QQQ", benchmark_spy)
    sector_symbol = SECTOR_BENCHMARKS.get(ticker, "SPY")
    benchmark_sector = timestamp_lookup.get(sector_symbol, benchmark_spy)
    current_ts = current.timestamp
    lookback_5_ts = price_rows[current_index - 5].timestamp if current_index >= 5 else None
    lookback_20_ts = price_rows[current_index - 20].timestamp if current_index >= 20 else None

    base_return_5d = _price_return(price_rows, current_index, 5)
    base_return_20d = _price_return(price_rows, current_index, 20)
    spy_return_5d = _reference_return(benchmark_spy, current_ts, lookback_5_ts)
    spy_return_20d = _reference_return(benchmark_spy, current_ts, lookback_20_ts)
    qqq_return_5d = _reference_return(benchmark_qqq, current_ts, lookback_5_ts)
    sector_return_5d = _reference_return(benchmark_sector, current_ts, lookback_5_ts)
    sector_return_20d = _reference_return(benchmark_sector, current_ts, lookback_20_ts)

    base_series_20, spy_series_20 = _aligned_return_series(price_rows, current_index, benchmark_spy, 20)
    _, sector_series_20 = _aligned_return_series(price_rows, current_index, benchmark_sector, 20)

    rel_to_spy_return_5d = base_return_5d - spy_return_5d if base_return_5d is not None and spy_return_5d is not None else None
    rel_to_spy_return_20d = base_return_20d - spy_return_20d if base_return_20d is not None and spy_return_20d is not None else None
    rel_to_qqq_return_5d = base_return_5d - qqq_return_5d if base_return_5d is not None and qqq_return_5d is not None else None
    rel_to_sector_return_5d = base_return_5d - sector_return_5d if base_return_5d is not None and sector_return_5d is not None else None
    rel_to_sector_return_20d = base_return_20d - sector_return_20d if base_return_20d is not None and sector_return_20d is not None else None

    lookback_timestamps = [row.timestamp for row in price_rows[max(0, current_index - 19): current_index + 1]]
    return {
        "price_close": close_price,
        "return_1d": next_return,
        "price_return_2d": _price_return(price_rows, current_index, 2),
        "price_return_3d": _price_return(price_rows, current_index, 3),
        "price_return_5d": base_return_5d,
        "price_return_10d": _price_return(price_rows, current_index, 10),
        "price_return_15d": _price_return(price_rows, current_index, 15),
        "price_return_20d": base_return_20d,
        "ma_gap_5d": ma_gap_5d,
        "ma_gap_20d": ma_gap_20d,
        "ma_crossover_5_20": ma_crossover_5_20,
        "ema_gap_12d": ema_gap_12d,
        "ema_gap_26d": ema_gap_26d,
        "ema_crossover_12_26": ema_crossover_12_26,
        "range_pct_1d": range_pct_1d,
        "atr_14_pct": atr_14_pct,
        "rolling_volatility_20d": rolling_volatility_20d,
        "downside_volatility_20d": downside_volatility_20d,
        "volatility_regime_60d": volatility_regime_60d,
        "volume_zscore_20d": volume_zscore_20d,
        "volume_change_5d": volume_change_5d,
        "volume_ratio_20d": volume_ratio_20d,
        "rsi_5": _rsi(price_rows, current_index, 5),
        "rsi_14": _rsi(price_rows, current_index, 14),
        "bollinger_z_20": bollinger_z_20,
        "breakout_20d": breakout_20d,
        "drawdown_20d": drawdown_20d,
        "trend_slope_10d": _linear_slope(close_window_10),
        "trend_slope_20d": _linear_slope(close_window_20),
        "up_day_ratio_10d": up_day_ratio_10d,
        "rel_to_spy_return_5d": rel_to_spy_return_5d,
        "rel_to_spy_return_20d": rel_to_spy_return_20d,
        "rel_to_qqq_return_5d": rel_to_qqq_return_5d,
        "rel_to_sector_return_5d": rel_to_sector_return_5d,
        "rel_to_sector_return_20d": rel_to_sector_return_20d,
        "market_beta_20d": _beta(base_series_20, spy_series_20),
        "market_corr_20d": _correlation(base_series_20, spy_series_20),
        "sector_corr_20d": _correlation(base_series_20, sector_series_20),
        "volume_vs_spy_ratio_20d": _volume_ratio_against_reference(current.volume, benchmark_spy, lookback_timestamps),
    }



def _previous_feature_state(
    ticker: str,
    rows: list[MarketPrice],
    current_index: int,
    existing_map: dict[tuple[str, datetime, int], FeatureSnapshot],
    computed_states: dict[tuple[str, datetime], dict[str, float | None]],
    window_hours: int = 24,
) -> dict[str, float | None]:
    if current_index <= 0:
        return {"news_count": None, "sentiment_mean": None}

    prev_window_end = rows[current_index - 1].timestamp
    if (ticker, prev_window_end) in computed_states:
        return computed_states[(ticker, prev_window_end)]

    existing = existing_map.get((ticker, prev_window_end, window_hours))
    if not existing:
        return {"news_count": None, "sentiment_mean": None}
    return {"news_count": float(existing.news_count), "sentiment_mean": existing.sentiment_mean}


def _needs_feature_backfill(row: FeatureSnapshot) -> bool:
    return any(getattr(row, field) is None for field in PRICE_BACKFILL_FIELDS + [
        "max_relevance",
        "source_weight_mean",
        "weighted_news_count",
        "weighted_sentiment_sum",
        "positive_news_ratio",
        "negative_news_ratio",
        "source_diversity",
        "news_count_72h",
        "event_intensity",
        "news_count_change_24h",
        "sentiment_momentum_24h",
    ])


def _should_refresh_feature_row(row: FeatureSnapshot, in_window: list[NewsSignal]) -> bool:
    if _needs_feature_backfill(row):
        return True
    if not in_window:
        return False
    if row.created_at is None:
        return True
    return any(signal.created_at and signal.created_at > row.created_at for signal in in_window)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ratio(selected: list[float], total: list[float]) -> float | None:
    if not total:
        return None
    return len(selected) / len(total)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5

