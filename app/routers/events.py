"""
Batched event ingestion. The frontend tracker queues events client-side
and flushes them together — this endpoint accepts a batch in one request
rather than one request per click. After ingesting, checks the cheap
trigger condition and, if crossed, kicks off the full agent run as a
FastAPI background task — the request returns immediately either way,
the agent (if triggered) runs after the response is already sent.
"""
from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.runner import run_agent_for_user
from app.db.session import get_db
from app.models.event import Event
from app.models.user import User
from app.routers.deps import get_current_user
from app.schemas.event import EventBatchIn
from app.services import trigger_service

router = APIRouter(prefix="/api/events", tags=["events"])


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def ingest_events(
    payload: EventBatchIn,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    rows = [
        Event(
            user_id=user.id,
            event_type=e.event_type,
            product_id=e.product_id,
            event_metadata=e.event_metadata,
        )
        for e in payload.events
    ]
    db.add_all(rows)
    await db.commit()

    should_run, reason = await trigger_service.should_trigger(db, user.id)
    await db.commit()  # persist cognitive-model row if it was just created

    if should_run:
        background_tasks.add_task(run_agent_for_user, user.id, reason)

    return {"ingested": len(rows), "agent_triggered": should_run, "trigger_reason": reason}