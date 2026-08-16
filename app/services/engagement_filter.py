"""
Deterministic per-user engagement exclusion for candidate selection.

Products the user has already engaged with must never be recommended again:

    - checked out      (CHECKOUT_START — product_ids live in event_metadata,
                         never assume product_id holds the course ID)
    - added to cart    (ADD_TO_CART)
    - previously recommended  (any Recommendation row for this user)
    - viewed           (PRODUCT_VIEW within the last RECENT_VIEW_WINDOW_HOURS)

Applied AFTER candidate extraction in retrieve, on every retrieval attempt
(filtered and retry alike), so a user's newest interest can surface instead
of re-serving courses they already acted on.
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.recommendation import Recommendation

RECENT_VIEW_WINDOW_HOURS = 24


def _aware(dt: datetime) -> datetime:
    """SQLite doesn't preserve tzinfo across a round-trip — normalize to
    aware UTC before comparing with datetime.now(timezone.utc)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def get_excluded_product_ids(user_id: str) -> set[str]:
    """All product IDs the user has already engaged with and must not be
    recommended. Deterministic, per-user, no schema or event changes."""
    excluded: set[str] = set()
    view_cutoff = datetime.now(timezone.utc) - timedelta(hours=RECENT_VIEW_WINDOW_HOURS)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Event).where(
                Event.user_id == user_id,
                Event.event_type.in_([
                    EventType.ADD_TO_CART,
                    EventType.CHECKOUT_START,
                    EventType.PRODUCT_VIEW,
                ]),
            )
        )
        for event in result.scalars().all():
            if event.event_type == EventType.PRODUCT_VIEW:
                if _aware(event.created_at) < view_cutoff:
                    continue  # only recent views exclude
            if event.event_type == EventType.CHECKOUT_START:
                metadata = event.event_metadata or {}
                ids = metadata.get("product_ids")
                if isinstance(ids, list) and ids:
                    excluded.update(str(i) for i in ids)
                    continue
                if event.product_id:
                    excluded.add(event.product_id)
                    continue
            if event.product_id:
                excluded.add(event.product_id)

        recs = await db.execute(
            select(Recommendation).where(Recommendation.user_id == user_id)
        )
        for rec in recs.scalars().all():
            for product in rec.products or []:
                pid = product.get("product_id") if isinstance(product, dict) else None
                if pid:
                    excluded.add(str(pid))

    return excluded