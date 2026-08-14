"""Daily digest scheduler logic: only users with activity in the lookback
window get a fresh recommendation; stale users are skipped."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
from app.services.scheduler import run_daily_digest
from sqlalchemy import select


@pytest.mark.asyncio
async def test_daily_digest_targets_only_recently_active_users(fake_llm):
    async with AsyncSessionLocal() as db:
        active = User(email=f"active-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        stale = User(email=f"stale-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add_all([active, stale])
        await db.flush()

        db.add(Event(user_id=active.id, event_type=EventType.SEARCH, event_metadata={"query": "recent"}))
        stale_event = Event(user_id=stale.id, event_type=EventType.SEARCH, event_metadata={"query": "old"})
        stale_event.created_at = datetime.now(timezone.utc) - timedelta(days=3)
        db.add(stale_event)
        await db.commit()
        active_id, stale_id = active.id, stale.id

    await run_daily_digest()

    async with AsyncSessionLocal() as db:
        active_rec = (await db.execute(
            select(Recommendation).where(Recommendation.user_id == active_id)
        )).scalar_one_or_none()
        stale_rec = (await db.execute(
            select(Recommendation).where(Recommendation.user_id == stale_id)
        )).scalar_one_or_none()
        assert active_rec is not None
        assert stale_rec is None