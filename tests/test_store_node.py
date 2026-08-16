"""store node — persists the agent's output and deactivates the previous
active recommendation so history is preserved but only the latest surfaces."""
import pytest

from app.agent.nodes.store import store_node
from app.db.session import AsyncSessionLocal
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
from sqlalchemy import select


@pytest.mark.asyncio
async def test_store_persists_and_deactivates_previous():
    async with AsyncSessionLocal() as db:
        user = User(email="store-test@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.flush()
        db.add(Recommendation(user_id=user.id, narrative="old", is_active=True,
                              products=[{"product_id": "p1", "reason": "r"}]))
        await db.commit()
        user_id = user.id

    await store_node({"user_id": user_id, "narrative": "new", "trigger_reason": "test",
                      "recommended_products": [{"product_id": "p2", "reason": "r2"}],
                      "persuasion_strategy": "curiosity_framing", "confidence": 0.8,
                      "reasoning_chain": ["step"], "alternatives_considered": []})

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Recommendation).where(Recommendation.user_id == user_id))).scalars().all()
        assert len(rows) == 2  # history preserved
        active = [r for r in rows if r.is_active]
        assert len(active) == 1
        assert active[0].narrative == "new"
        assert active[0].trigger_reason == "test"
        assert active[0].products == [{"product_id": "p2", "reason": "r2"}]
        assert not [r for r in rows if r.narrative == "old"][0].is_active


@pytest.mark.asyncio
async def test_empty_result_keeps_previous_recommendation_active():
    """A run with zero valid candidates must NOT overwrite a good existing
    recommendation with an empty one — the old recommendation stays active
    until a genuinely valid new set exists."""
    async with AsyncSessionLocal() as db:
        user = User(email="store-guard@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.flush()
        db.add(Recommendation(user_id=user.id, narrative="good-old", is_active=True,
                              products=[{"product_id": "p1", "reason": "r"}]))
        await db.commit()
        user_id = user.id

    await store_node({"user_id": user_id, "narrative": "fallback",
                      "recommended_products": [],
                      "persuasion_strategy": "none", "confidence": 0.0,
                      "reasoning_chain": [], "alternatives_considered": []})

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Recommendation).where(Recommendation.user_id == user_id))).scalars().all()
        assert len(rows) == 1  # no empty row was written
        assert rows[0].is_active
        assert rows[0].narrative == "good-old"  # previous recommendation preserved