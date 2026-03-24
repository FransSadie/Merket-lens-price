import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import get_settings
from app.ingestion.pipeline import run_all_ingestion

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def start_scheduler() -> None:
    settings = get_settings()
    minutes = settings.ingest_interval_minutes
    scheduler.add_job(
        run_all_ingestion,
        trigger=IntervalTrigger(minutes=minutes),
        id="ingestion_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info("Scheduler started with %s minute interval", minutes)


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

