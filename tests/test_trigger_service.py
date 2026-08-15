"""Trigger service: deterministic new-event selection (no timestamp-offset
ambiguity) and the count-based cursor used to decide when to run the agent."""
import uuid
from datetime import datetime, timezone

import pytest

from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.user import User
from app.security import hash_password
from app.services import trigger_service
from sqlalchemy import text


async def _make_user() -> str:
    async with AsyncSessionLocal() as db:
        user = User(email=f"t-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.commit()
        return user.id


async def _add_events(user_id: str, count: int, ts: datetime | None = None) -> list[str]:
    """Insert events with explicit, deterministic ids so tie-breaking is
    observable in the test."""
    ids = []
    async with AsyncSessionLocal() as db:
        for i in range(count):
            event = Event(
                user_id=user_id,
                event_type=EventType.SEARCH,
                event_metadata={"query": f"q{i}"},
            )
            event.id = f"00000000-0000-0000-0000-{i:012d}"
            if ts is not None:
                event.created_at = ts
            db.add(event)
            ids.append(event.id)
        await db.commit()
    return ids


@pytest.mark.asyncio
async def test_fetch_new_events_is_deterministic_with_tied_timestamps():
    """Events sharing an identical created_at must still be selected
    unambiguously (tie broken by id) — the old timestamp-offset approach
    could silently pick the wrong rows here."""
    user_id = await _make_user()
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = await _add_events(user_id, 3, ts=ts)

    async with AsyncSessionLocal() as db:
        newest = await trigger_service._fetch_new_events(db, user_id, 1)
        assert [r.id for r in newest] == [ids[2]]  # highest (created_at, id)

        last_two = await trigger_service._fetch_new_events(db, user_id, 2)
        assert [r.id for r in last_two] == [ids[1], ids[2]]  # chronological order preserved

        assert await trigger_service._fetch_new_events(db, user_id, 0) == []


@pytest.mark.asyncio
async def test_fetch_new_events_respects_created_at_then_id():
    """A later timestamp wins even with a lower id; within one timestamp,
    the higher id wins."""
    user_id = await _make_user()
    async with AsyncSessionLocal() as db:
        older = Event(user_id=user_id, event_type=EventType.SEARCH, event_metadata={"query": "old"})
        older.id = "00000000-0000-0000-0000-000000000005"
        older.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        newer = Event(user_id=user_id, event_type=EventType.SEARCH, event_metadata={"query": "new"})
        newer.id = "00000000-0000-0000-0000-000000000001"
        newer.created_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
        db.add_all([older, newer])
        await db.commit()

    async with AsyncSessionLocal() as db:
        newest = await trigger_service._fetch_new_events(db, user_id, 1)
        assert [r.id for r in newest] == [newer.id]


@pytest.mark.asyncio
async def test_should_trigger_uses_count_cursor():
    """After the agent has processed N events, only the events beyond N are
    considered new — and they're the actual newest rows, not an offset guess."""
    user_id = await _make_user()
    await _add_events(user_id, 5)

    async with AsyncSessionLocal() as db:
        await trigger_service.get_or_create_cognitive_model(db, user_id)
        await db.commit()
        # Pretend 3 events were already processed. A raw UPDATE bypasses the
        # ORM's updated_at onupdate (which would otherwise re-arm the cooldown
        # by stamping now) so the backdated timestamp survives the commit.
        await db.execute(
            text(
                "UPDATE user_cognitive_models "
                "SET last_event_count_at_update = 3, updated_at = :t "
                "WHERE user_id = :u"
            ),
            {"t": datetime(2000, 1, 1, tzinfo=timezone.utc), "u": user_id},
        )
        await db.commit()

    async with AsyncSessionLocal() as db:
        ok, reason = await trigger_service.should_trigger(db, user_id)

    assert ok is True
    assert "2 new events" in reason