"""Full agent graph end-to-end (offline, mocked LLM): behavior → cognitive
model → retrieval → evaluation → filtering → generation → persistence.
Verifies the stored recommendation is grounded in the real catalog."""
import json
import uuid

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


async def _seed_catalog_and_user():
    async with AsyncSessionLocal() as db:
        products = []
        for title, desc, cat, price in [
            ("Agentic AI Fundamentals", "Build reasoning agents with LangGraph and RAG.", "AI", 49.99),
            ("Production RAG at Scale", "Advanced retrieval-augmented generation patterns.", "AI", 79.99),
            ("Introduction to Baking", "Bread and pastries at home.", "Culinary", 19.99),
        ]:
            p = Product(title=title, description=desc, category=cat, price=price, vector_id=str(uuid.uuid4()))
            db.add(p)
            await db.flush()
            await vector_store.upsert_product(
                vector_id=p.vector_id, embedding=fake_embedding(p.to_embedding_text()),
                document=p.to_embedding_text(),
                metadata={"category": p.category, "price": p.price, "sql_id": p.id},
            )
            products.append(p)

        user = User(email=f"e2e-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.flush()
        db.add_all([
            Event(user_id=user.id, event_type=EventType.SEARCH, event_metadata={"query": "agentic ai"}),
            Event(user_id=user.id, event_type=EventType.PRODUCT_VIEW, product_id=products[0].id,
                  event_metadata={"title": products[0].title, "category": "AI"}),
            Event(user_id=user.id, event_type=EventType.SEARCH, event_metadata={"query": "langgraph rag"}),
        ])
        await db.commit()
        return user.id


@pytest.mark.asyncio
async def test_full_graph_runs_and_persists_grounded_recommendation(fake_llm):
    user_id = await _seed_catalog_and_user()

    final_state = await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "pytest e2e"})

    # every pipeline stage ran
    assert final_state.get("cognitive_model")
    assert final_state.get("narrative")
    assert final_state.get("confidence") is not None
    assert final_state.get("persuasion_strategy")

    # all recommended products are real catalog products (grounding contract)
    async with AsyncSessionLocal() as db:
        stored = (await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )).scalar_one_or_none()
        assert stored is not None
        assert stored.narrative == final_state["narrative"]
        assert stored.trigger_reason == "pytest e2e"

        ids = {p["product_id"] for p in stored.products}
        db_ids = set((await db.execute(select(Product.id))).scalars().all())
        assert ids <= db_ids  # nothing invented

    # mocked LLM made a bounded number of calls (model_user, evaluate,
    # generate, reflect — no runaway regeneration loop)
    assert fake_llm.chat_calls <= 8


@pytest.mark.asyncio
async def test_agent_is_idempotent_across_runs(fake_llm):
    """Re-running the agent for the same user must deactivate the previous
    recommendation and never duplicate the active one."""
    user_id = await _seed_catalog_and_user()
    await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "run 1"})
    await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "run 2"})

    async with AsyncSessionLocal() as db:
        active = (await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )).scalars().all()
        assert len(active) == 1
        assert active[0].trigger_reason == "run 2"