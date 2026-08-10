"""
Decides whether enough new user activity has accumulated to justify
running the agent again — the mechanism behind "don't call the LLM on
every single event." Cheap to evaluate (just a row count), so it's safe
to call after every event batch without itself being wasteful.
"""
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_model import UserCognitiveModel
from app.models.event import Event

# Fire the agent once at least this many new events have landed since the
# cognitive model was last updated. Tune this — lower = more responsive but
# more LLM calls; higher = cheaper but staler recommendations.
EVENT_THRESHOLD = 3


async def get_or_create_cognitive_model(db: AsyncSession, user_id: str) -> UserCognitiveModel:
    result = await db.execute(
        select(UserCognitiveModel).where(UserCognitiveModel.user_id == user_id)
    )
    model = result.scalar_one_or_none()
    if model is None:
        model = UserCognitiveModel(user_id=user_id, last_event_count_at_update=0)
        db.add(model)
        await db.flush()
    return model


async def should_trigger(db: AsyncSession, user_id: str) -> tuple[bool, str]:
    """Returns (should_run, reason). Reason is stored on the eventual
    Recommendation row (trigger_reason) so it's visible why the agent ran
    when it did — useful for debugging and for the "why am I seeing this"
    transparency angle."""
    total_events_result = await db.execute(
        select(func.count()).select_from(Event).where(Event.user_id == user_id)
    )
    total_events = total_events_result.scalar_one()

    model = await get_or_create_cognitive_model(db, user_id)
    new_events = total_events - model.last_event_count_at_update

    if new_events >= EVENT_THRESHOLD:
        return True, f"{new_events} new events since last update (threshold: {EVENT_THRESHOLD})"

    return False, f"only {new_events} new events, below threshold of {EVENT_THRESHOLD}"