"""
Read-only aggregations for the admin console (Overview / Users /
Observability). Nothing here mutates the database, and everything returned
traces back to actual rows. The recommendation backend is intentionally
frozen — this module only reads existing tables and derives summaries.

Key derivation notes:
- "Agent run" = a temporal cluster of LLM call log rows. The pipeline has no
  run_id column (schema is frozen), but every run makes its LLM calls within
  a few seconds, and the run-trigger cooldown guarantees runs for the same
  user are >= 60s apart, so a 120s gap between consecutive calls is a safe
  cluster boundary. A recommendation written at the end of a run falls
  within 120s of the cluster's last call and is matched 1:1 when unique.
- "Estimated cost" = real token counts (from provider usage) priced at
  documented blended per-1M-token rates. An estimate, clearly labelled, not
  a billable figure; mock calls contribute $0.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.cognitive_model import UserCognitiveModel
from app.models.event import Event, EventType
from app.models.llm_call_log import LLMCallLog
from app.models.product import Product, SyncStatus
from app.models.recommendation import Recommendation
from app.models.user import User

# Documented blended rates used ONLY for the cost estimate (real tokens x rate).
CHAT_PROMPT_RATE_PER_1M = 0.30        # USD per 1M prompt tokens
CHAT_COMPLETION_RATE_PER_1M = 1.20    # USD per 1M completion tokens
EMBEDDING_RATE_PER_1M = 0.02          # USD per 1M tokens
COST_BASIS = "estimated from real tokens at blended rates (prompt $0.30/1M, completion $1.20/1M, embedding $0.02/1M); mock calls cost $0"

# A run's LLM calls land within seconds of each other; anything farther apart
# than this is a different run.
RUN_CLUSTER_GAP_SECONDS = 120
# A stored recommendation is written by the store node immediately after the
# run's final LLM call — match when a recommendation lands in this window.
RECOMMENDATION_MATCH_WINDOW_SECONDS = 120


def _aware(dt: datetime | None) -> datetime | None:
    """SQLite round-trips DateTime(timezone=True) columns as naive; normalize."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _enum_val(x) -> str:
    return x.value if isinstance(x, Enum) else str(x)


def estimate_call_cost(call: LLMCallLog) -> float:
    """Estimated spend for one call. Mock calls (dev-mode embeddings) cost $0."""
    if call.is_mock:
        return 0.0
    if call.call_type == "embedding":
        tokens = call.total_tokens or 0
        return tokens / 1_000_000 * EMBEDDING_RATE_PER_1M
    prompt = call.prompt_tokens or 0
    completion = call.completion_tokens or 0
    return (prompt * CHAT_PROMPT_RATE_PER_1M + completion * CHAT_COMPLETION_RATE_PER_1M) / 1_000_000


def _cost_summary(calls: list[LLMCallLog]) -> dict:
    total_tokens = sum(c.total_tokens or 0 for c in calls)
    estimated_usd = round(sum(estimate_call_cost(c) for c in calls), 6)
    return {"total_tokens": total_tokens, "estimated_usd": estimated_usd, "basis": COST_BASIS}


def cluster_calls(calls: list[LLMCallLog], gap_seconds: int = RUN_CLUSTER_GAP_SECONDS) -> list[list[LLMCallLog]]:
    """Group LLM calls into pipeline runs by time proximity."""
    ordered = sorted(calls, key=lambda c: _aware(c.created_at))
    clusters: list[list[LLMCallLog]] = []
    current: list[LLMCallLog] = []
    prev_ts: datetime | None = None
    for call in ordered:
        ts = _aware(call.created_at)
        if prev_ts is not None and (ts - prev_ts).total_seconds() > gap_seconds:
            clusters.append(current)
            current = []
        current.append(call)
        prev_ts = ts
    if current:
        clusters.append(current)
    return clusters


async def _match_recommendations(db: AsyncSession, clusters: list[list[LLMCallLog]], window_start: datetime) -> dict[int, Recommendation]:
    """Best-effort link of a run cluster to the recommendation its store node
    wrote. Only attaches when exactly one recommendation lands in the window
    after the cluster's final call, so a match is never guessed."""
    if not clusters:
        return {}
    recs_result = await db.execute(
        select(Recommendation)
        .options(selectinload(Recommendation.user))
        .where(Recommendation.created_at >= window_start - timedelta(seconds=60))
    )
    recs = recs_result.scalars().all()
    matched: dict[int, Recommendation] = {}
    for i, cluster in enumerate(clusters):
        end = max(_aware(c.created_at) for c in cluster)
        candidates = [
            r for r in recs
            if end <= _aware(r.created_at) <= end + timedelta(seconds=RECOMMENDATION_MATCH_WINDOW_SECONDS)
        ]
        if len(candidates) == 1:
            matched[i] = candidates[0]
    return matched


def _summarize_cluster(cluster: list[LLMCallLog], recommendation: Recommendation | None = None) -> dict:
    times = [_aware(c.created_at) for c in cluster]
    start = min(times)
    end = max(times)
    failed = [c for c in cluster if not c.success]

    rec = None
    if recommendation is not None:
        rec = {
            "id": recommendation.id,
            "user_email": recommendation.user.email if recommendation.user else "Not recorded",
            "trigger_reason": recommendation.trigger_reason,
            "persuasion_strategy": recommendation.persuasion_strategy,
            "confidence": recommendation.confidence,
            "product_count": len(recommendation.products or []),
            "created_at": _aware(recommendation.created_at),
        }

    return {
        "start_time": start,
        "end_time": end,
        "duration_seconds": round((end - start).total_seconds(), 2),
        "llm_calls": len(cluster),
        "chat_calls": sum(1 for c in cluster if c.call_type == "chat"),
        "embedding_calls": sum(1 for c in cluster if c.call_type == "embedding"),
        "models": sorted({c.model for c in cluster}),
        "total_tokens": sum(c.total_tokens or 0 for c in cluster),
        "estimated_cost_usd": round(sum(estimate_call_cost(c) for c in cluster), 6),
        "success": all(c.success for c in cluster),
        "error": failed[0].error if failed else None,
        "recommendation": rec,
    }


def _event_summary(event: Event) -> str:
    meta = event.event_metadata or {}
    if event.event_type == EventType.SEARCH:
        q = meta.get("query") or meta.get("q") or ""
        return f"searched {q!r}" if q else "searched"
    if event.event_type == EventType.PRODUCT_VIEW:
        return "viewed a product"
    if event.event_type == EventType.CLICK:
        target = meta.get("target") or ""
        return f"clicked {target}" if target else "clicked"
    if event.event_type == EventType.TIME_SPENT:
        seconds = meta.get("seconds") or 0
        return f"spent {seconds}s on page"
    if event.event_type == EventType.ADD_TO_CART:
        return "added to cart"
    if event.event_type == EventType.CHECKOUT_START:
        return "started checkout"
    return "viewed a page"


async def _product_titles(db: AsyncSession, product_ids: set[str]) -> dict[str, str]:
    if not product_ids:
        return {}
    result = await db.execute(select(Product).where(Product.id.in_(product_ids)))
    return {p.id: p.title for p in result.scalars().all()}


async def _recent_activity(db: AsyncSession, limit: int = 12) -> list[dict]:
    result = await db.execute(
        select(Event).options(selectinload(Event.user)).order_by(Event.created_at.desc()).limit(limit)
    )
    events = list(result.scalars().all())
    titles = await _product_titles(db, {e.product_id for e in events if e.product_id})
    return [
        {
            "id": e.id,
            "user_email": e.user.email if e.user else "unknown",
            "event_type": _enum_val(e.event_type),
            "product_title": titles.get(e.product_id) if e.product_id else None,
            "summary": _event_summary(e),
            "created_at": _aware(e.created_at),
        }
        for e in events
    ]


async def build_overview(db: AsyncSession, window_hours: int = 24) -> dict:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=window_hours)

    total_users = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    active_users = (await db.execute(
        select(func.count(func.distinct(Event.user_id))).where(Event.created_at >= since)
    )).scalar_one()
    recommendations_24h = (await db.execute(
        select(func.count()).select_from(Recommendation).where(Recommendation.created_at >= since)
    )).scalar_one()

    calls = list((await db.execute(
        select(LLMCallLog).where(LLMCallLog.created_at >= since).order_by(LLMCallLog.created_at)
    )).scalars().all())

    real_calls = [c for c in calls if not c.is_mock]
    successful = [c for c in calls if c.success]
    avg_latency = round(sum(c.latency_ms for c in calls) / len(calls), 1) if calls else 0.0
    success_rate = round(len(successful) / len(calls), 3) if calls else 1.0

    clusters = cluster_calls(calls)
    matched = await _match_recommendations(db, clusters, since)
    run_summaries = [_summarize_cluster(cl, matched.get(i)) for i, cl in enumerate(clusters)]
    runs_with_rec = sum(1 for s in run_summaries if s["recommendation"] is not None)
    avg_pipeline = (
        round(sum(s["duration_seconds"] for s in run_summaries) / len(run_summaries), 1)
        if run_summaries else 0.0
    )

    # Vector store health — from the real dual-write status on products.
    sync_rows = (await db.execute(
        select(Product.sync_status, func.count()).group_by(Product.sync_status)
    )).all()
    sync = {"total": 0, "synced": 0, "pending": 0, "failed": 0}
    for status, count in sync_rows:
        key = _enum_val(status)
        if key in sync:
            sync[key] += count
    sync["total"] = sync["synced"] + sync["pending"] + sync["failed"]
    failed_products = (await db.execute(
        select(Product)
        .where(Product.sync_status == SyncStatus.FAILED)
        .order_by(Product.updated_at.desc())
        .limit(5)
    )).scalars().all()
    sync["sample_errors"] = [
        {"id": p.id, "title": p.title, "error": p.sync_error or "unknown"} for p in failed_products
    ]

    recent_activity = await _recent_activity(db, limit=12)

    # Behavioral profile aggregate — real UserCognitiveModel rows.
    cog_models = list((await db.execute(select(UserCognitiveModel))).scalars().all())
    decision_dist: Counter = Counter()
    price_dist: Counter = Counter()
    top_intents: Counter = Counter()
    top_categories: Counter = Counter()
    readiness_sum = 0.0
    for m in cog_models:
        decision_dist[_enum_val(m.decision_stage)] += 1
        price_dist[_enum_val(m.price_sensitivity)] += 1
        readiness_sum += m.purchase_readiness or 0.0
        top_intents.update(m.inferred_intents or [])
        top_categories.update(m.category_affinity or [])

    return {
        "generated_at": now,
        "window_hours": window_hours,
        "total_users": total_users,
        "active_users_24h": active_users,
        "model_calls_24h": len(calls),
        "real_calls_24h": len(real_calls),
        "mock_calls_24h": len(calls) - len(real_calls),
        "success_rate_24h": success_rate,
        "avg_latency_ms_24h": avg_latency,
        "recommendations_24h": recommendations_24h,
        "agent_runs": {
            "runs_24h": len(run_summaries),
            "runs_with_recommendation": runs_with_rec,
            "avg_pipeline_seconds": avg_pipeline,
            "llm_calls_24h": sum(s["llm_calls"] for s in run_summaries),
            "total_tokens_24h": sum(s["total_tokens"] for s in run_summaries),
            "estimated_cost_usd_24h": round(sum(s["estimated_cost_usd"] for s in run_summaries), 6),
        },
        "cost": _cost_summary(calls),
        "vector_sync": sync,
        "recent_activity": recent_activity,
        "decision_stage_distribution": dict(decision_dist),
        "price_sensitivity_distribution": dict(price_dist),
        "avg_purchase_readiness": round(readiness_sum / len(cog_models), 2) if cog_models else 0.0,
        "cognitive_models_count": len(cog_models),
        "top_inferred_intents": [{"name": k, "count": v} for k, v in top_intents.most_common(8)],
        "top_category_affinity": [{"name": k, "count": v} for k, v in top_categories.most_common(8)],
    }


async def list_users(db: AsyncSession, limit: int = 100, offset: int = 0) -> dict:
    total = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    users = list((await db.execute(
        select(User).order_by(User.created_at.desc()).offset(offset).limit(limit)
    )).scalars().all())
    if not users:
        return {"total": total, "limit": limit, "offset": offset, "items": []}

    user_ids = [u.id for u in users]

    event_counts = dict((await db.execute(
        select(Event.user_id, func.count()).where(Event.user_id.in_(user_ids)).group_by(Event.user_id)
    )).all())
    last_events = {
        uid: _aware(dt)
        for uid, dt in (await db.execute(
            select(Event.user_id, func.max(Event.created_at))
            .where(Event.user_id.in_(user_ids))
            .group_by(Event.user_id)
        )).all()
    }
    rec_counts = dict((await db.execute(
        select(Recommendation.user_id, func.count())
        .where(Recommendation.user_id.in_(user_ids))
        .group_by(Recommendation.user_id)
    )).all())
    feedback: dict[str, dict[str, int]] = {}
    for uid, fb, n in (await db.execute(
        select(Recommendation.user_id, Recommendation.feedback, func.count())
        .where(Recommendation.user_id.in_(user_ids))
        .group_by(Recommendation.user_id, Recommendation.feedback)
    )).all():
        bucket = feedback.setdefault(uid, {"up": 0, "down": 0})
        if fb == "up":
            bucket["up"] = n
        elif fb == "down":
            bucket["down"] = n

    active_by_user: dict[str, Recommendation] = {}
    for r in (await db.execute(
        select(Recommendation)
        .where(Recommendation.user_id.in_(user_ids), Recommendation.is_active.is_(True))
        .order_by(Recommendation.created_at.desc())
    )).scalars().all():
        active_by_user.setdefault(r.user_id, r)

    cog_by_user = {
        m.user_id: m
        for m in (await db.execute(select(UserCognitiveModel))).scalars().all()
        if m.user_id in user_ids
    }

    items = []
    for u in users:
        cog = cog_by_user.get(u.id)
        act = active_by_user.get(u.id)
        items.append({
            "id": u.id,
            "email": u.email,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "role": _enum_val(u.role),
            "created_at": _aware(u.created_at),
            "events_count": event_counts.get(u.id, 0),
            "recommendations_count": rec_counts.get(u.id, 0),
            "last_event_at": last_events.get(u.id),
            "feedback_up": feedback.get(u.id, {}).get("up", 0),
            "feedback_down": feedback.get(u.id, {}).get("down", 0),
            "has_active_recommendation": act is not None,
            "active_trigger_reason": act.trigger_reason if act else None,
            "active_confidence": act.confidence if act else None,
            "decision_stage": _enum_val(cog.decision_stage) if cog else None,
            "purchase_readiness": cog.purchase_readiness if cog else None,
            "price_sensitivity": _enum_val(cog.price_sensitivity) if cog else None,
            "top_inferred_intents": (cog.inferred_intents or [])[:5] if cog else [],
        })
    return {"total": total, "limit": limit, "offset": offset, "items": items}


async def get_user_detail(db: AsyncSession, user_id: str) -> dict | None:
    user = await db.get(User, user_id)
    if user is None:
        return None

    event_count = (await db.execute(
        select(func.count()).select_from(Event).where(Event.user_id == user_id)
    )).scalar_one()
    rec_count = (await db.execute(
        select(func.count()).select_from(Recommendation).where(Recommendation.user_id == user_id)
    )).scalar_one()
    fb_rows = (await db.execute(
        select(Recommendation.feedback, func.count())
        .where(Recommendation.user_id == user_id)
        .group_by(Recommendation.feedback)
    )).all()
    feedback_up = sum(n for fb, n in fb_rows if fb == "up")
    feedback_down = sum(n for fb, n in fb_rows if fb == "down")

    cog = (await db.execute(
        select(UserCognitiveModel).where(UserCognitiveModel.user_id == user_id)
    )).scalar_one_or_none()

    recs = list((await db.execute(
        select(Recommendation).where(Recommendation.user_id == user_id)
        .order_by(Recommendation.created_at.desc()).limit(20)
    )).scalars().all())
    rec_briefs = [
        {
            "id": r.id,
            "created_at": _aware(r.created_at),
            "is_active": r.is_active,
            "feedback": r.feedback,
            "trigger_reason": r.trigger_reason,
            "confidence": r.confidence,
            "persuasion_strategy": r.persuasion_strategy,
            "product_count": len(r.products or []),
            "narrative": r.narrative,
        }
        for r in recs
    ]

    events = list((await db.execute(
        select(Event).where(Event.user_id == user_id).order_by(Event.created_at.desc()).limit(12)
    )).scalars().all())
    titles = await _product_titles(db, {e.product_id for e in events if e.product_id})
    ev_briefs = [
        {
            "id": e.id,
            "event_type": _enum_val(e.event_type),
            "product_title": titles.get(e.product_id) if e.product_id else None,
            "summary": _event_summary(e),
            "created_at": _aware(e.created_at),
        }
        for e in events
    ]

    profile = None
    if cog is not None:
        profile = {
            "stated_intents": cog.stated_intents or [],
            "inferred_intents": cog.inferred_intents or [],
            "decision_stage": _enum_val(cog.decision_stage),
            "purchase_readiness": cog.purchase_readiness or 0.0,
            "price_sensitivity": _enum_val(cog.price_sensitivity),
            "detected_objections": cog.detected_objections or [],
            "brand_affinity": cog.brand_affinity or [],
            "category_affinity": cog.category_affinity or [],
            "recent_searches": cog.recent_searches or [],
            "recent_categories": cog.recent_categories or [],
            "session_arc": cog.session_arc or "",
            "updated_at": _aware(cog.updated_at),
        }

    return {
        "id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "role": _enum_val(user.role),
        "created_at": _aware(user.created_at),
        "events_count": event_count,
        "recommendations_count": rec_count,
        "feedback_up": feedback_up,
        "feedback_down": feedback_down,
        "cognitive_profile": profile,
        "recommendations": rec_briefs,
        "recent_events": ev_briefs,
    }


async def build_observability(db: AsyncSession, window_hours: int = 24, max_runs: int = 50) -> dict:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=window_hours)

    calls = list((await db.execute(
        select(LLMCallLog).where(LLMCallLog.created_at >= since).order_by(LLMCallLog.created_at)
    )).scalars().all())

    real_calls = [c for c in calls if not c.is_mock]
    successful = [c for c in calls if c.success]
    avg_latency = round(sum(c.latency_ms for c in calls) / len(calls), 1) if calls else 0.0
    success_rate = round(len(successful) / len(calls), 3) if calls else 1.0

    model_breakdown: dict[str, int] = {}
    for c in calls:
        model_breakdown[c.model] = model_breakdown.get(c.model, 0) + 1

    clusters = cluster_calls(calls)
    matched = await _match_recommendations(db, clusters, since)
    runs = [_summarize_cluster(cl, matched.get(i)) for i, cl in enumerate(clusters)][:max_runs]

    recent_calls = [
        {
            "id": c.id,
            "call_type": c.call_type,
            "model": c.model,
            "is_mock": c.is_mock,
            "latency_ms": c.latency_ms,
            "total_tokens": c.total_tokens,
            "success": c.success,
            "error": c.error,
            "created_at": _aware(c.created_at),
            "estimated_cost_usd": round(estimate_call_cost(c), 6),
        }
        for c in reversed(calls[-25:])
    ]

    return {
        "generated_at": now,
        "window_hours": window_hours,
        "calls_24h": len(calls),
        "real_calls_24h": len(real_calls),
        "mock_calls_24h": len(calls) - len(real_calls),
        "success_rate_24h": success_rate,
        "avg_latency_ms_24h": avg_latency,
        "total_tokens_24h": sum(c.total_tokens or 0 for c in calls),
        "estimated_cost_usd_24h": round(sum(estimate_call_cost(c) for c in calls), 6),
        "model_breakdown": model_breakdown,
        "runs": runs,
        "recent_calls": recent_calls,
    }