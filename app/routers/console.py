"""
Agent Console — real operational data, no fabricated numbers. Every value
returned here is computed from actual LLMCallLog rows. Admin-only, since
this exposes internal operational detail (error messages, model names).
"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.llm_call_log import LLMCallLog
from app.routers.deps import require_admin
from app.schemas.console import ConsoleStats

router = APIRouter(prefix="/api/console", tags=["console"])


@router.get("/stats", response_model=ConsoleStats)
async def get_console_stats(db: AsyncSession = Depends(get_db), _admin=Depends(require_admin)):
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    result = await db.execute(
        select(LLMCallLog).where(LLMCallLog.created_at >= since).order_by(LLMCallLog.created_at.desc())
    )
    calls = result.scalars().all()

    real_calls = [c for c in calls if not c.is_mock]
    successful = [c for c in calls if c.success]

    model_breakdown: dict[str, int] = {}
    for c in calls:
        model_breakdown[c.model] = model_breakdown.get(c.model, 0) + 1

    avg_latency = sum(c.latency_ms for c in calls) / len(calls) if calls else 0.0
    success_rate = len(successful) / len(calls) if calls else 1.0

    return ConsoleStats(
        calls_today=len(calls),
        real_calls_today=len(real_calls),
        mock_calls_today=len(calls) - len(real_calls),
        avg_latency_ms=round(avg_latency, 1),
        success_rate=round(success_rate, 3),
        model_breakdown=model_breakdown,
        recent_calls=calls[:25],
    )