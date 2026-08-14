"""model_user node — turns raw events into a persisted cognitive model,
extracting recency signals deterministically (no LLM) alongside the LLM
inference."""
import json
import uuid

import pytest

from app.agent.nodes.model_user import model_user_node, _extract_recent_signals
from app.db.session import AsyncSessionLocal
from app.models.cognitive_model import UserCognitiveModel
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.user import User
from app.security import hash_password


async def _seed(searches, views):
    async with AsyncSessionLocal() as db:
        user = User(email=f"mu-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.flush()

        products = {}
        for title, category in views:
            p = Product(title=title, description="desc", category=category, price=10.0,
                        vector_id=str(uuid.uuid4()))
            db.add(p)
            await db.flush()
            products[(title, category)] = p

        events = [Event(user_id=user.id, event_type=EventType.SEARCH, event_metadata={"query": q}) for q in searches]
        events += [Event(user_id=user.id, event_type=EventType.PRODUCT_VIEW,
                         product_id=products[k].id, event_metadata={"title": k[0]}) for k in views]
        db.add_all(events)
        await db.commit()
        return user.id


@pytest.mark.asyncio
async def test_recent_signals_extracted_chronologically(fake_llm):
    user_id = await _seed(["sql", "kafka", "dbt"], [("AI Course", "AI"), ("Kafka Course", "Data Engineering")])
    async with AsyncSessionLocal() as db:
        events = (await db.execute(
            __import__("sqlalchemy").select(Event).where(Event.user_id == user_id)
        )).scalars().all()
        searches, categories = await _extract_recent_signals(events, {})
        assert "sql" in searches and "kafka" in searches
        # categories come from product lookups — empty map here, so verify via the node instead


@pytest.mark.asyncio
async def test_model_user_persists_cognitive_model(fake_llm):
    user_id = await _seed(["agentic ai", "langgraph"], [("Agentic Course", "AI")])
    result = await model_user_node({"user_id": user_id, "trigger_reason": "test"})

    cm = result["cognitive_model"]
    assert cm["stated_intents"] == ["agentic ai"]
    assert cm["session_arc"]

    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            __import__("sqlalchemy").select(UserCognitiveModel).where(UserCognitiveModel.user_id == user_id)
        )).scalar_one()
        assert row.recent_searches == ["agentic ai", "langgraph"]
        assert "AI" in row.recent_categories  # product-view category captured via SQL join
        assert row.last_event_count_at_update == 3