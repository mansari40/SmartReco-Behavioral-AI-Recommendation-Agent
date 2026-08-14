"""
Recommendation retrieval + feedback. The agent that populates this table
runs independently (triggered by events or the daily digest) — this
router just serves what's stored and records user feedback on it.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.recommendation import Recommendation
from app.models.user import User
from app.routers.deps import get_current_user
from app.schemas.recommendation import FeedbackIn, RecommendationOut

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


@router.post("/{recommendation_id}/feedback", response_model=RecommendationOut)
async def submit_feedback(
    recommendation_id: str,
    payload: FeedbackIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if payload.feedback not in ("up", "down"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "feedback must be 'up' or 'down'")

    recommendation = await db.get(Recommendation, recommendation_id)
    if not recommendation or recommendation.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Recommendation not found")

    recommendation.feedback = payload.feedback
    await db.commit()
    await db.refresh(recommendation)
    return recommendation