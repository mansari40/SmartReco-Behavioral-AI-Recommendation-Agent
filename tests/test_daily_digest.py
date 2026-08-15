"""Daily digest scheduler logic: only users with activity in the lookback
window get a fresh recommendation; stale users are skipped; users who just
received a recommendation (within the TTL) with no strong new signal are
skipped too, mirroring the event trigger."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
from app.services.scheduler import run_daily_digest, _should_digest_user
from sqlalchemy import select


async def _make_user() -> str:
    async with AsyncSessionLocal() as db:
        user = User(email=f"u-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.commit()
        return user.id


async def _add_rec(user_id: str, minutes_ago: int) -> None:
    async with AsyncSessionLocal() as db:
        rec = Recommendation(user_id=user_id, narrative="rec", is_active=True)
        rec.created_at = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        db.add(rec)
        await db.commit()


async def _add_event(user_id: str, event_type: EventType, minutes_ago: int) -> None:
    async with AsyncSessionLocal() as db:
        event = Event(user_id=user_id, event_type=event_type, event_metadata={})
        event.created_at = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        db.add(event)
        await db.commit()


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


@pytest.mark.asyncio
async def test_should_digest_user_skips_fresh_rec_with_weak_signal():
    """A recommendation < TTL old with only weak events after it blocks the
    digest — mirrors the event trigger's fresh-recommendation gate."""
    user_id = await _make_user()
    await _add_event(user_id, EventType.PAGE_VIEW, minutes_ago=10)
    await _add_rec(user_id, minutes_ago=5)

    assert await _should_digest_user(user_id) is False


@pytest.mark.asyncio
async def test_should_digest_user_runs_fresh_rec_with_strong_signal():
    """A strong new signal (search) after a fresh recommendation must pass
    through, exactly like the event trigger."""
    user_id = await _make_user()
    await _add_event(user_id, EventType.SEARCH, minutes_ago=10)
    await _add_rec(user_id, minutes_ago=5)
    await _add_event(user_id, EventType.SEARCH, minutes_ago=1)

    assert await _should_digest_user(user_id) is True


@pytest.mark.asyncio
async def test_should_digest_user_runs_when_rec_stale():
    """A recommendation older than the TTL no longer blocks the digest."""
    user_id = await _make_user()
    await _add_event(user_id, EventType.PAGE_VIEW, minutes_ago=60)
    await _add_rec(user_id, minutes_ago=120)

    assert await _should_digest_user(user_id) is True


@pytest.mark.asyncio
async def test_should_digest_user_runs_without_recommendation():
    """No recommendation at all -> always digest."""
    user_id = await _make_user()
    await _add_event(user_id, EventType.PAGE_VIEW, minutes_ago=1)

    assert await _should_digest_user(user_id) is True


@pytest.mark.asyncio
async def test_daily_digest_skips_fresh_rec_user(fake_llm):
    """End-to-end: a user who already has a fresh active recommendation and
    only weak new events is not run by the digest (no duplicate LLM spend)."""
    async with AsyncSessionLocal() as db:
        fresh = User(email=f"fresh-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add(fresh)
        await db.flush()
        fresh_rec = Recommendation(user_id=fresh.id, narrative="recent", is_active=True)
        fresh_rec.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        db.add(fresh_rec)
        weak_event = Event(user_id=fresh.id, event_type=EventType.PAGE_VIEW, event_metadata={})
        weak_event.created_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.add(weak_event)
        await db.commit()
        fresh_id = fresh.id

    await run_daily_digest()

    async with AsyncSessionLocal() as db:
        recs = (await db.execute(
            select(Recommendation).where(Recommendation.user_id == fresh_id)
        )).scalars().all()
        assert len(recs) == 1, "digest should not have produced a second recommendation"
        assert recs[0].narrative == "recent"