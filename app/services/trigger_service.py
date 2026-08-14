"""
Decides whether enough new user activity has accumulated to justify
running the agent again — the mechanism behind "don't call the LLM on
every single event." Cheap to evaluate (row count + one timestamp
comparison), so it's safe to call after every event batch without
itself being wasteful.
"""
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_model import UserCognitiveModel
from app.models.event import Event, EventType

EVENT_THRESHOLD = 3
MIN_SECONDS_BETWEEN_RUNS = 60  # cooldown — prevents rapid-fire re-triggering while testing
MIN_SCORE_THRESHOLD = 3.0
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


async def should_trigger(db: AsyncSession, user_id: str) -> tuple[bool, str]:
    total_events_result = await db.execute(
        select(func.count()).select_from(Event).where(Event.user_id == user_id)
    )
    total_events = total_events_result.scalar_one()

    model = await get_or_create_cognitive_model(db, user_id)
    new_events = total_events - model.last_event_count_at_update

    if new_events <= 0:
        return False, "no new events since last agent update"

    query = (
        select(Event)
        .where(Event.user_id == user_id)
        .order_by(Event.created_at)
        .offset(model.last_event_count_at_update)
    )
    events_result = await db.execute(query)
    new_event_rows = events_result.scalars().all()
    activity_score = _score_event_batch(new_event_rows)

    if activity_score < MIN_SCORE_THRESHOLD and new_events < EVENT_THRESHOLD:
        return False, (
            f"{new_events} new events with weak signal ({activity_score:.1f}), "
            f"below thresholds of {EVENT_THRESHOLD} events or {MIN_SCORE_THRESHOLD} score"
        )

    seconds_since_last = (datetime.now(timezone.utc) - _aware(model.updated_at)).total_seconds()
    if seconds_since_last < MIN_SECONDS_BETWEEN_RUNS:
        return False, f"cooldown active ({int(MIN_SECONDS_BETWEEN_RUNS - seconds_since_last)}s remaining)"

    return True, (
        f"triggered on {new_events} new events with aggregate score {activity_score:.1f} "
        f"(threshold {MIN_SCORE_THRESHOLD})"
    )