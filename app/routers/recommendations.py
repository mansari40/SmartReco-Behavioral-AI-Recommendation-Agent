"""
Recommendation retrieval endpoint. The agent that populates this table
(reasoning over user activity -> retrieve -> generate) is the next build
stage — this router just serves whatever the agent has already stored,
and is deliberately thin.
"""
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.recommendation import Recommendation
from app.models.user import User
from app.routers.deps import get_current_user
from app.schemas.recommendation import RecommendationOut

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.get("/me", response_model=RecommendationOut | None)
async def get_my_recommendation(
    db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Recommendation)
        .where(Recommendation.user_id == user.id, Recommendation.is_active.is_(True))
        .order_by(Recommendation.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()