import logging
from datetime import datetime, timedelta

from sqlalchemy import select

from app.db.models import FeatureSnapshot, IngestionRun, MarketLabel, MarketPrice, NewsArticle, NewsSignal
from app.db.session import get_db_session

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
            prev_tracker = _latest_feature_tracker(latest_rows)
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

                (
                    close_price,
                    return_1d,
                    price_return_3d,
                    price_return_5d,
                    price_return_10d,
                    price_return_20d,
                    ma_gap_5d,
                    ma_gap_20d,
                    ma_crossover_5_20,
                    range_pct_1d,
                    atr_14_pct,
                    rolling_volatility_20d,
                    volatility_regime_60d,
                    volume_zscore_20d,
                    volume_change_5d,
                ) = _price_features(rows, idx)
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
                row.price_return_3d = price_return_3d
                row.price_return_5d = price_return_5d
                row.price_return_10d = price_return_10d
                row.price_return_20d = price_return_20d
                row.ma_gap_5d = ma_gap_5d
                row.ma_gap_20d = ma_gap_20d
                row.ma_crossover_5_20 = ma_crossover_5_20
                row.range_pct_1d = range_pct_1d
                row.atr_14_pct = atr_14_pct
                row.rolling_volatility_20d = rolling_volatility_20d
                row.volatility_regime_60d = volatility_regime_60d
                row.volume_zscore_20d = volume_zscore_20d
                row.volume_change_5d = volume_change_5d
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
    for row in stale_rows:
        if _needs_feature_backfill(row):
            target_keys.add((row.ticker, row.window_end))

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


def _price_features(
    price_rows: list[MarketPrice], current_index: int
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    if not price_rows:
        return (None,) * 15
    if current_index < 0 or current_index >= len(price_rows):
        return (None,) * 15

    current = price_rows[current_index]
    close_price = current.close

    def price_return(days: int) -> float | None:
        if close_price is None or close_price == 0 or current_index < days:
            return None
        prev_close = price_rows[current_index - days].close
        if prev_close is None or prev_close == 0:
            return None
        return (close_price - prev_close) / prev_close

    next_return = None
    if close_price is not None and close_price != 0 and current_index + 1 < len(price_rows):
        next_close = price_rows[current_index + 1].close
        if next_close is not None:
            next_return = (next_close - close_price) / close_price

    price_return_3d = price_return(3)
    price_return_5d = price_return(5)
    price_return_10d = price_return(10)
    price_return_20d = price_return(20)

    ma_5 = _window_mean([row.close for row in price_rows[max(0, current_index - 4): current_index + 1]])
    ma_20 = _window_mean([row.close for row in price_rows[max(0, current_index - 19): current_index + 1]])
    ma_gap_5d = ((close_price - ma_5) / ma_5) if close_price is not None and ma_5 not in (None, 0) else None
    ma_gap_20d = ((close_price - ma_20) / ma_20) if close_price is not None and ma_20 not in (None, 0) else None
    ma_crossover_5_20 = ((ma_5 - ma_20) / ma_20) if ma_5 not in (None, 0) and ma_20 not in (None, 0) else None

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

    rolling_volatility_20d = None
    return_history = []
    if current_index >= 20:
        for i in range(current_index - 19, current_index + 1):
            prev = price_rows[i - 1].close if i - 1 >= 0 else None
            curr = price_rows[i].close
            if prev is not None and prev != 0 and curr is not None:
                return_history.append((curr - prev) / prev)
        rolling_volatility_20d = _std(return_history) if return_history else None

    vol_history = []
    if current_index >= 40:
        for end in range(current_index - 19, current_index + 1):
            sub_returns = []
            for i in range(max(1, end - 19), end + 1):
                prev = price_rows[i - 1].close
                curr = price_rows[i].close
                if prev is not None and prev != 0 and curr is not None:
                    sub_returns.append((curr - prev) / prev)
            if sub_returns:
                vol_history.append(_std(sub_returns))
    volatility_regime_60d = None
    if rolling_volatility_20d is not None and len([v for v in vol_history if v is not None]) >= 5:
        clean_vols = [v for v in vol_history if v is not None]
        avg_vol = _mean(clean_vols)
        sd_vol = _std(clean_vols)
        if avg_vol is not None and sd_vol is not None and sd_vol > 0:
            volatility_regime_60d = (rolling_volatility_20d - avg_vol) / sd_vol

    volume_zscore_20d = None
    volume_change_5d = None
    if current_index >= 20 and current.volume is not None:
        volumes = [price_rows[i].volume for i in range(current_index - 19, current_index + 1)]
        clean_volumes = [v for v in volumes if v is not None]
        if len(clean_volumes) >= 5:
            avg = _mean(clean_volumes)
            sd = _std(clean_volumes)
            if avg is not None and sd is not None and sd > 0:
                volume_zscore_20d = (current.volume - avg) / sd
    if current.volume is not None and current_index >= 5:
        prev_volume = price_rows[current_index - 5].volume
        if prev_volume not in (None, 0):
            volume_change_5d = (current.volume - prev_volume) / prev_volume

    return (
        close_price,
        next_return,
        price_return_3d,
        price_return_5d,
        price_return_10d,
        price_return_20d,
        ma_gap_5d,
        ma_gap_20d,
        ma_crossover_5_20,
        range_pct_1d,
        atr_14_pct,
        rolling_volatility_20d,
        volatility_regime_60d,
        volume_zscore_20d,
        volume_change_5d,
    )


def _latest_feature_tracker(rows: list[FeatureSnapshot]) -> dict[str, dict[str, float | None]]:
    tracker: dict[str, dict[str, float | None]] = {}
    for row in rows:
        if row.ticker not in tracker:
            tracker[row.ticker] = {"news_count": float(row.news_count), "sentiment_mean": row.sentiment_mean}
    return tracker


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
    return any(
        value is None
        for value in [
            row.max_relevance,
            row.source_weight_mean,
            row.weighted_news_count,
            row.weighted_sentiment_sum,
            row.positive_news_ratio,
            row.negative_news_ratio,
            row.source_diversity,
            row.news_count_72h,
            row.event_intensity,
            row.news_count_change_24h,
            row.sentiment_momentum_24h,
            row.price_return_3d,
            row.price_return_5d,
            row.price_return_10d,
            row.price_return_20d,
            row.ma_gap_5d,
            row.ma_gap_20d,
            row.ma_crossover_5_20,
            row.range_pct_1d,
            row.atr_14_pct,
            row.rolling_volatility_20d,
            row.volatility_regime_60d,
            row.volume_zscore_20d,
            row.volume_change_5d,
        ]
    )


def _should_refresh_feature_row(row: FeatureSnapshot, in_window: list[NewsSignal]) -> bool:
    if _needs_feature_backfill(row):
        return True
    if not in_window:
        return False
    if row.created_at is None:
        return True
    return any(signal.created_at and signal.created_at > row.created_at for signal in in_window)


def _window_mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


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

