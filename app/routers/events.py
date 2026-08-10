"""
Batched event ingestion. The frontend tracker (built later) queues events
client-side and flushes them together — this endpoint accepts a batch in
one request rather than one request per click, which is the actual
mechanism behind "non-blocking, efficient tracking."
"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

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


    return {"ingested": len(rows), "agent_would_trigger": should_run, "trigger_reason": reason}