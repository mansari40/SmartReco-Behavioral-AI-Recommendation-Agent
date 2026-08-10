"""
APScheduler-based daily digest: runs the recommendation agent for every
user who's had activity in the last 24 hours, independent of any HTTP
request. This is a real scheduled process (not a manual button, not tied
to request/response), started at app startup and stopped at shutdown.
"""
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from app.agent.runner import run_agent_for_user
from app.db.session import AsyncSessionLocal
from app.models.event import Event

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

# Runs once a day at 16:00 server time — "an email in the afternoon
# recapping the morning's interests," per the brief's own example.
DIGEST_HOUR = 16
LOOKBACK_HOURS = 24


async def run_daily_digest() -> None:
    logger.info("Daily digest job starting")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Event.user_id).where(Event.created_at >= cutoff).distinct()
        )
        active_user_ids = [row[0] for row in result.all()]

    logger.info("Daily digest: %d users with activity in last %dh", len(active_user_ids), LOOKBACK_HOURS)

    for user_id in active_user_ids:
        try:
            await run_agent_for_user(user_id, trigger_reason="daily digest (scheduled)")
        except Exception:
            logger.exception("Daily digest failed for user=%s", user_id)

    logger.info("Daily digest job complete")


def start_scheduler() -> None:
    scheduler.add_job(
        run_daily_digest,
        trigger=CronTrigger(hour=DIGEST_HOUR, minute=0),
        id="daily_digest",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started — daily digest set for %02d:00", DIGEST_HOUR)


def stop_scheduler() -> None:
    scheduler.shutdown(wait=False)