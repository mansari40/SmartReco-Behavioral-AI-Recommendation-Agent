"""Regression: the stale "For You" recommendations scenario.

User explores Data Engineering courses -> carts 2 -> checks out -> the agent
recommends NEW (unused) courses -> user shifts to AI courses, carts/checks
out -> another agent run.

Acceptance (CONTENT, not just re-runs):
- the new run reflects the NEWER AI interest (newest category leads)
- previously checked-out DE courses never reappear
- previously checked-out AI courses never reappear
- previously recommended courses never reappear
- recently viewed courses never reappear
- score-~0 candidates are never force-filled
- exclusions survive every retry attempt (newest -> recent -> unfiltered)
- retry progression is newest category -> broader recent categories ->
  unfiltered, never a blind jump
"""
import uuid
from datetime import datetime, timedelta

import pytest

from app.agent.graph import agent_graph
from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
from app.services import vector_store
from sqlalchemy import select
from tests.conftest import fake_embedding

DE1 = "Data Quality Engineering"
DE2 = "Data Warehousing Fundamentals"
DE3 = "dbt Analytics Engineering"
AI1 = "AI Agents Fundamentals"
AI2 = "Multimodal AI"
AI3 = "Production RAG at Scale"

Q1 = f"{DE3}. Category: Data Engineering. dbt transformations and testing."
Q2 = f"{AI2}. Category: AI. multimodal model integration and evaluation."


async def _seed_catalog():
    """5 real SQL products. The vector store is faked per-test, so no
    embeddings are upserted here — grounding still runs against SQL."""
    async with AsyncSessionLocal() as db:
        rows = []
        for title, desc, cat in [
            (DE1, "quality validation and monitoring", "Data Engineering"),
            (DE2, "warehouse modeling and ETL", "Data Engineering"),
            (DE3, "dbt transformations and testing", "Data Engineering"),
            (AI1, "agent construction and tool use", "AI"),
            (AI2, "multimodal model integration", "AI"),
            (AI3, "advanced retrieval-augmented generation", "AI"),
        ]:
            p = Product(title=title, description=desc, category=cat, price=49.99,
                        vector_id=str(uuid.uuid4()))
            db.add(p)
            await db.flush()
            rows.append(p)
        await db.commit()
        return {p.title: p.id for p in rows}


def _raw(pool):
    """Raw vector-store result shape expected by _extract_candidates:
    pool is a list of (product_id, distance)."""
    ids, metadatas, distances = [], [], []
    for pid, distance in pool:
        ids.append(pid)
        metadatas.append({"sql_id": pid, "category": "x", "price": 49.99})
        distances.append(distance)
    return {"ids": [ids], "metadatas": [metadatas], "distances": [distances]}


def _make_fake_query(ids_by_title, embedding_1, embedding_2):
    """Deterministic fake vector store. Distances are keyed on the run's
    query embedding (run 1 = DE-oriented query, run 2 = AI-oriented query)
    and the metadata filter stage, mirroring real semantic behavior:
    an AI-oriented query ranks AI courses first and vice versa."""
    de = {ids_by_title[DE1]: 0.5, ids_by_title[DE2]: 0.5, ids_by_title[DE3]: 0.45}
    ai = {ids_by_title[AI1]: 0.98, ids_by_title[AI2]: 0.99, ids_by_title[AI3]: 0.99}
    ai_near = {ids_by_title[AI1]: 0.3, ids_by_title[AI2]: 0.1, ids_by_title[AI3]: 0.1}
    de_far = {ids_by_title[DE1]: 0.7, ids_by_title[DE2]: 0.7, ids_by_title[DE3]: 0.6}

    async def fake_query(embedding, top_k=10, where=None):
        if embedding == embedding_1:  # run 1: DE-oriented query
            if where == {"category": "Data Engineering"}:
                pool = list(de.items())
            elif where == {"category": {"$in": ["Data Engineering"]}}:
                pool = list(de.items())
            elif where == {"category": {"$in": ["Data Engineering", "AI"]}}:
                pool = list(de.items()) + list(ai.items())
            else:  # unfiltered fallback
                pool = list(de.items()) + list(ai.items())
        else:  # run 2: AI-oriented query
            if where == {"category": "AI"}:
                pool = list(ai_near.items())
            elif where == {"category": {"$in": ["Data Engineering", "AI"]}}:
                pool = list(de_far.items()) + list(ai_near.items())
            else:  # unfiltered fallback
                pool = list(de_far.items()) + list(ai_near.items())
        return _raw(pool)

    return fake_query


async def _seed_user(ids_by_title):
    async with AsyncSessionLocal() as db:
        user = User(email=f"stale-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.flush()
        await db.commit()
        user_id = user.id

    base = datetime(2026, 8, 16, 12, 0)
    await _add_events(user_id, ids_by_title, phase=1, base=base)
    return user_id


async def _add_events(user_id, ids_by_title, phase, base):
    """Append behavior events for a phase to an EXISTING user. Phase 1 = DE
    exploration -> cart -> checkout; phase 2 = AI exploration -> cart ->
    checkout. Timestamps are explicit so ordering is deterministic."""
    async with AsyncSessionLocal() as db:
        events = []
        if phase == 1:
            events += [
                (EventType.PRODUCT_VIEW, ids_by_title[DE1], {}),
                (EventType.PRODUCT_VIEW, ids_by_title[DE2], {}),
                (EventType.ADD_TO_CART, ids_by_title[DE1], {}),
                (EventType.ADD_TO_CART, ids_by_title[DE2], {}),
                (EventType.CHECKOUT_START, None, {"product_ids": [ids_by_title[DE1], ids_by_title[DE2]]}),
            ]
        else:
            events += [
                (EventType.PRODUCT_VIEW, ids_by_title[AI1], {}),
                (EventType.PRODUCT_VIEW, ids_by_title[AI2], {}),
                (EventType.ADD_TO_CART, ids_by_title[AI1], {}),
                (EventType.CHECKOUT_START, None, {"product_ids": [ids_by_title[AI1]]}),
            ]
        for i, (event_type, product_id, metadata) in enumerate(events):
            db.add(Event(user_id=user_id, event_type=event_type, product_id=product_id,
                         event_metadata=metadata,
                         created_at=base + timedelta(minutes=i // 4, seconds=(i % 4) * 5)))
        await db.commit()


async def _active_rec(user_id):
    async with AsyncSessionLocal() as db:
        rec = (await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )).scalar_one_or_none()
        return rec


@pytest.mark.asyncio
async def test_stale_recommendation_scenario_shifts_to_newest_interest(fake_llm, monkeypatch):
    ids_by_title = await _seed_catalog()
    user_id = await _seed_user(ids_by_title)

    from app.agent.nodes import retrieve as retrieve_mod

    queries = iter([Q1, Q2])
    monkeypatch.setattr(retrieve_mod, "_build_query_text", lambda cm: next(queries))

    fake_query = _make_fake_query(ids_by_title, fake_embedding(Q1), fake_embedding(Q2))
    calls = []

    async def recording_query(embedding, top_k=10, where=None):
        calls.append(where)
        return await fake_query(embedding, top_k=top_k, where=where)

    monkeypatch.setattr(vector_store, "query", recording_query)

    # ---- run 1: DE exploration -> cart -> checkout -> recommendation ----
    await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "run 1"})
    calls_run1 = list(calls)

    rec1 = await _active_rec(user_id)
    assert rec1 is not None
    products1 = {p["product_id"] for p in rec1.products}
    assert products1 == {ids_by_title[DE3]}  # a NEW DE course — not the engaged ones
    assert ids_by_title[DE1] not in products1  # viewed + carted + checked out
    assert ids_by_title[DE2] not in products1  # viewed + carted + checked out
    for p in rec1.products:
        assert p["score"] >= 0.2  # relevance floor honored, nothing force-filled

    # ---- run 2: AI exploration -> cart -> checkout -> new recommendation ----
    await _add_events(user_id, ids_by_title, phase=2, base=datetime(2026, 8, 16, 12, 10))
    calls.clear()
    await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "run 2"})

    rec2 = await _active_rec(user_id)
    assert rec2 is not None
    products2 = {p["product_id"] for p in rec2.products}
    # THE acceptance criterion: the set is now based on the NEWER AI interest
    assert products2 == {ids_by_title[AI3]}
    # ...and nothing engaged or previously recommended reappears
    for engaged in (DE1, DE2, DE3, AI1, AI2):
        assert ids_by_title[engaged] not in products2
    for p in rec2.products:
        assert p["score"] >= 0.2

    # previous set was replaced, not duplicated
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Recommendation).where(Recommendation.user_id == user_id))).scalars().all()
        assert len(rows) == 2
        assert sum(1 for r in rows if r.is_active) == 1

    # ---- retry progression: newest category -> recent categories -> unfiltered ----
    # run 1 recent categories = ["Data Engineering"] (single -> scalar filter)
    assert calls_run1 == [
        {"category": "Data Engineering"},
        {"category": "Data Engineering"},
        None,
    ]
    # run 2 recent categories = ["Data Engineering", "AI"] — attempt 1 MUST lead
    # with the NEWEST category (AI), never the historical DE dominance
    assert calls == [
        {"category": "AI"},
        {"category": {"$in": ["Data Engineering", "AI"]}},
        None,
    ]

    # exclusions survived every attempt: run 2's unfiltered pool still only
    # surfaced the one genuinely unused course (AI3) — verified by rec2 == {AI3}
    assert fake_llm.chat_calls == 4  # model_user + generate per run
    assert fake_llm.embedding_calls == 2  # one embedding per run, reused across retries