"""Behavioral event ingestion: batching, per-user isolation, and the
intelligent trigger gate (don't run the agent after every event)."""
import pytest

from tests.conftest import auth

from app.db.session import AsyncSessionLocal
from app.models.event import Event
from sqlalchemy import func, select


async def _ingest(client, token, events):
    resp = await client.post("/api/events", json={"events": events}, headers=auth(token))
    assert resp.status_code == 202
    return resp.json()


async def test_batch_ingest_stores_all_events(client, user_token):
    result = await _ingest(client, user_token, [
        {"event_type": "page_view", "event_metadata": {"path": "/"}},
        {"event_type": "search", "event_metadata": {"query": "RAG"}},
        {"event_type": "time_spent", "event_metadata": {"seconds": 42}},
    ])
    assert result["ingested"] == 3

    async with AsyncSessionLocal() as db:
        count = (await db.execute(select(func.count()).select_from(Event))).scalar_one()
        assert count == 3


async def test_events_require_auth(client):
    resp = await client.post("/api/events", json={"events": []})
    assert resp.status_code in (401, 422)


async def test_user_isolation(client, user_token):
    """A user must never see another user's events."""
    token2 = None
    import uuid as _uuid
    resp = await client.post("/api/auth/register", json={
        "email": f"other-{_uuid.uuid4().hex[:8]}@test.com",
        "first_name": "O", "last_name": "U", "password": "pass1234",
    })
    resp = await client.post("/api/auth/login", data={"username": resp.json()["email"], "password": "pass1234"})
    token2 = resp.json()["access_token"]

    await _ingest(client, user_token, [{"event_type": "search", "event_metadata": {"query": "private interest"}}])
    await _ingest(client, token2, [{"event_type": "search", "event_metadata": {"query": "other interest"}}])

    recent = (await client.get("/api/events/recent", headers=auth(user_token))).json()
    queries = [e["event_metadata"].get("query") for e in recent]
    assert "private interest" in queries
    assert "other interest" not in queries


async def test_weak_activity_does_not_trigger_agent(client, user_token):
    """A single low-weight page view should not fire the agent."""
    result = await _ingest(client, user_token, [
        {"event_type": "page_view", "event_metadata": {"path": "/"}},
    ])
    assert result["agent_triggered"] is False
    assert "below thresholds" in result["trigger_reason"]


async def test_strong_activity_triggers_agent(client, user_token):
    """Searches and product views are high-weight — should trigger."""
    result = await _ingest(client, user_token, [
        {"event_type": "search", "event_metadata": {"query": "LangGraph"}},
        {"event_type": "search", "event_metadata": {"query": "RAG"}},
        {"event_type": "product_view", "event_metadata": {"title": "Agentic AI", "category": "AI"}},
    ])
    assert result["agent_triggered"] is True


async def test_cooldown_blocks_rapid_retrigger(client, user_token):
    """The second burst inside the cooldown window must not re-run.
    Tested at the trigger-service level for determinism: the endpoint runs
    the agent as an async background task whose completion timing would
    otherwise race the assertion."""
    from datetime import datetime, timezone

    from app.security import decode_access_token
    from app.services import trigger_service

    uid = decode_access_token(user_token)

    async with AsyncSessionLocal() as db:
        db.add_all([
            Event(user_id=uid, event_type="search", event_metadata={"query": "one"}),
            Event(user_id=uid, event_type="search", event_metadata={"query": "two"}),
            Event(user_id=uid, event_type="search", event_metadata={"query": "three"}),
        ])
        await db.commit()
        ok, reason = await trigger_service.should_trigger(db, uid)
        assert ok is True  # fresh model, backdated updated_at → first trigger allowed

    # simulate the agent run completing (model_user sets updated_at=now)
    async with AsyncSessionLocal() as db:
        model = await trigger_service.get_or_create_cognitive_model(db, uid)
        model.updated_at = datetime.now(timezone.utc)
        db.add(Event(user_id=uid, event_type="search", event_metadata={"query": "three"}))
        await db.commit()
        ok, reason = await trigger_service.should_trigger(db, uid)
        assert ok is False
        assert "cooldown" in reason


async def test_new_user_first_trigger_not_blocked(client, user_token):
    """Regression: the very first trigger for a fresh user used to always
    hit the cooldown because the cognitive model was created with
    updated_at=now."""
    result = await _ingest(client, user_token, [
        {"event_type": "search", "event_metadata": {"query": "first search"}},
        {"event_type": "search", "event_metadata": {"query": "second search"}},
        {"event_type": "search", "event_metadata": {"query": "third search"}},
    ])
    assert result["agent_triggered"] is True
