"""
Admin-only, read-only operational API. Powers the Overview / Users /
Observability pages. Nothing here mutates data — the recommendation backend
is intentionally frozen; this router only reads and aggregates existing rows.
All endpoints are guarded by `require_admin` (403 for non-admins).
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.routers.deps import require_admin
from app.schemas.admin import (
    ObservabilityData,
    OverviewData,
    UserDetailData,
    UsersPage,
)
from app.services import admin_stats

router = APIRouter(prefix="/api/admin", tags=["admin"])

DEFAULT_WINDOW_HOURS = 24


@router.get("/overview", response_model=OverviewData)
async def admin_overview(
    window_hours: int = Query(DEFAULT_WINDOW_HOURS, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
    _admin=Depends(require_admin),
):
    return await admin_stats.build_overview(db, window_hours=window_hours)


@router.get("/users", response_model=UsersPage)
async def admin_users(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _admin=Depends(require_admin),
):
    return await admin_stats.list_users(db, limit=limit, offset=offset)


@router.get("/users/{user_id}", response_model=UserDetailData)
async def admin_user_detail(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(require_admin),
):
    data = await admin_stats.get_user_detail(db, user_id)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return data


@router.get("/observability", response_model=ObservabilityData)
async def admin_observability(
    window_hours: int = Query(DEFAULT_WINDOW_HOURS, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
    _admin=Depends(require_admin),
):
    return await admin_stats.build_observability(db, window_hours=window_hours)