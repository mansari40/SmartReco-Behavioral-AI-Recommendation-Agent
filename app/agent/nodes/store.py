"""
store node: persists the agent's final output as a Recommendation row.
Marks any previous active recommendation for this user as inactive first,
so GET /api/recommendations/me always returns the newest one while
history is preserved (not deleted) for anyone who wants to look back.
"""
from sqlalchemy import select

from app.agent.state import AgentState
from app.db.session import AsyncSessionLocal
from app.models.recommendation import Recommendation


async def store_node(state: AgentState) -> dict:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == state["user_id"],
                Recommendation.is_active.is_(True),
            )
        )
        for old in result.scalars().all():
            old.is_active = False

        new_recommendation = Recommendation(
            user_id=state["user_id"],
            narrative=state.get("narrative", ""),
            products=state.get("recommended_products", []),
            persuasion_strategy=state.get("persuasion_strategy", ""),
            confidence=state.get("confidence", 0.0),
            reasoning_chain=state.get("reasoning_chain", []),
            alternatives_considered=state.get("alternatives_considered", []),
            trigger_reason=state.get("trigger_reason", ""),
            is_active=True,
        )
        db.add(new_recommendation)
        await db.commit()
        await db.refresh(new_recommendation)

        return {}