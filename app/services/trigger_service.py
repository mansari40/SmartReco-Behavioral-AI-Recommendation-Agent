"""
Decides whether enough new user activity has accumulated to justify
running the agent again — the mechanism behind "don't call the LLM on
every single event." Cheap to evaluate (row count + one timestamp
comparison), so it's safe to call after every event batch without
itself being wasteful.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_model import UserCognitiveModel
from app.models.event import Event, EventType
from app.models.recommendation import Recommendation

EVENT_THRESHOLD = 3
MIN_SECONDS_BETWEEN_RUNS = 60  # cooldown — prevents rapid-fire re-triggering while testing
MIN_SCORE_THRESHOLD = 3.0
# Skip re-running the agent for a user who just received a recommendation
# unless the new activity is a high-intent signal. Cuts repeated full
# pipeline runs (each ~3 LLM calls) after light browsing post-recommendation.
RECOMMENDATION_TTL_SECONDS = 1800  # 30 minutes
STRONG_SIGNAL_TYPES = {EventType.ADD_TO_CART, EventType.CHECKOUT_START, EventType.SEARCH}
EVENT_WEIGHTS = {
    EventType.PAGE_VIEW: 0.4,
    EventType.PRODUCT_VIEW: 1.5,
    EventType.SEARCH: 2.0,
    EventType.CLICK: 1.0,
    EventType.TIME_SPENT: 0.6,
    EventType.ADD_TO_CART: 3.0,
    EventType.CHECKOUT_START: 4.0,
}


def _score_event(event: Event) -> float:
    base = EVENT_WEIGHTS.get(event.event_type, 1.0)
    if event.event_type == EventType.TIME_SPENT:
        seconds = int(event.event_metadata.get("seconds", 0) or 0)
        return min(seconds / 60 * 0.75, 3.0)
    if event.event_type == EventType.SEARCH and event.event_metadata.get("query"):
        return base + 0.5
    return base


def _score_event_batch(events: list[Event]) -> float:
    score = 0.0
    repeated_views = Counter(
        e.product_id for e in events if e.event_type == EventType.PRODUCT_VIEW and e.product_id
    )
    for event in events:
        score += _score_event(event)

    for count in repeated_views.values():
        if count > 1:
            score += min((count - 1) * 0.5, 2.0)

    return score


def _aware(dt: datetime) -> datetime:
    """SQLite doesn't preserve tzinfo across a round-trip even when the
    column is declared DateTime(timezone=True) — a freshly-loaded model's
    updated_at comes back naive, while datetime.now(timezone.utc) is
    aware, and subtracting the two raises. Normalize before comparing."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def get_or_create_cognitive_model(db: AsyncSession, user_id: str) -> UserCognitiveModel:
    result = await db.execute(
        select(UserCognitiveModel).where(UserCognitiveModel.user_id == user_id)
    )
    model = result.scalar_one_or_none()
    if model is None:
        model = UserCognitiveModel(user_id=user_id, last_event_count_at_update=0)
        # Backdate so the cooldown never blocks a user's very first trigger:
        # a brand-new model has no "last run" to protect against.
        model.updated_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
        db.add(model)
        await db.flush()
    return model


async def _fresh_recommendation_blocks(db: AsyncSession, user_id: str, new_event_rows: list[Event]) -> tuple[bool, str]:
    """If the user received a recommendation within the TTL window and the
    new events carry no high-intent signal (search/add-to-cart/checkout),
    skip the run — the user just got a recommendation, re-running the full
    pipeline for a couple of page views wastes LLM calls. High-intent
    signals always pass through."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=RECOMMENDATION_TTL_SECONDS)
    result = await db.execute(
        select(Recommendation)
        .where(
            Recommendation.user_id == user_id,
            Recommendation.is_active.is_(True),
        )
        .order_by(Recommendation.created_at.desc())
        .limit(1)
    )
    rec = result.scalar_one_or_none()
    if rec is None or rec.created_at is None:
        return False, ""
    if (datetime.now(timezone.utc) - _aware(rec.created_at)).total_seconds() >= RECOMMENDATION_TTL_SECONDS:
        return False, ""
    if any(e.event_type in STRONG_SIGNAL_TYPES for e in new_event_rows):
        return False, ""
    return True, "fresh recommendation exists and no strong new signal"


async def _fetch_new_events(db: AsyncSession, user_id: str, new_events: int) -> list[Event]:
    """Deterministically return the `new_events` most-recent events that the
    agent hasn't seen yet.

    The old approach skipped `last_event_count_at_update` rows in a
    `created_at`-only ordering — ambiguous when events share the exact same
    timestamp, which could silently re-select already-processed events and
    miss the actually-new ones. The count is exact (the agent bumps
    `last_event_count_at_update` to the true total after every run, and
    events are never deleted), so we slice the newest N rows directly and
    break timestamp ties by id (a stable, deterministic UUID).
    """
    if new_events <= 0:
        return []
    result = await db.execute(
        select(Event)
        .where(Event.user_id == user_id)
        .order_by(Event.created_at.desc(), Event.id.desc())
        .limit(new_events)
    )
    rows = list(result.scalars().all())
    rows.reverse()  # back to chronological (oldest first) for downstream consumers
    return rows


async def should_trigger(db: AsyncSession, user_id: str) -> tuple[bool, str]:
    total_events_result = await db.execute(
        select(func.count()).select_from(Event).where(Event.user_id == user_id)
    )
    total_events = total_events_result.scalar_one()

    model = await get_or_create_cognitive_model(db, user_id)
    new_events = total_events - model.last_event_count_at_update

    if new_events <= 0:
        return False, "no new events since last agent update"

    new_event_rows = await _fetch_new_events(db, user_id, new_events)
    activity_score = _score_event_batch(new_event_rows)

    if activity_score < MIN_SCORE_THRESHOLD and new_events < EVENT_THRESHOLD:
        return False, (
            f"{new_events} new events with weak signal ({activity_score:.1f}), "
            f"below thresholds of {EVENT_THRESHOLD} events or {MIN_SCORE_THRESHOLD} score"
        )

    fresh_blocked, fresh_reason = await _fresh_recommendation_blocks(db, user_id, new_event_rows)
    if fresh_blocked:
        return False, fresh_reason

    seconds_since_last = (datetime.now(timezone.utc) - _aware(model.updated_at)).total_seconds()
    if seconds_since_last < MIN_SECONDS_BETWEEN_RUNS:
        return False, f"cooldown active ({int(MIN_SECONDS_BETWEEN_RUNS - seconds_since_last)}s remaining)"

    return True, (
        f"triggered on {new_events} new events with aggregate score {activity_score:.1f} "
        f"(threshold {MIN_SCORE_THRESHOLD})"
    )