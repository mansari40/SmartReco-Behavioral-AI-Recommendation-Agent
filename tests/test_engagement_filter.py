"""engagement_filter — deterministic per-user exclusions: checked out,
added to cart, previously recommended, and products viewed within the last
24 hours must never be recommended again."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
from app.services.engagement_filter import RECENT_VIEW_WINDOW_HOURS, get_excluded_product_ids


async def _seed_user(events, recommendation_products=()):
    async with AsyncSessionLocal() as db:
        user = User(email=f"eng-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.flush()
        for event in events:
            db.add(Event(user_id=user.id, **event))
        for product in recommendation_products:
            db.add(Recommendation(user_id=user.id, narrative="old", is_active=True,
                                  products=[{"product_id": product, "reason": "old"}]))
        await db.commit()
        return user.id


@pytest.mark.asyncio
async def test_checkout_exclusion_uses_metadata_product_ids():
    """checkout_start carries the courses in event_metadata.product_ids —
    never assume product_id holds the course ID."""
    user_id = await _seed_user([
        {"event_type": EventType.CHECKOUT_START,
         "event_metadata": {"product_ids": ["p-aaa", "p-bbb"], "count": 2}},
    ])
    excluded = await get_excluded_product_ids(user_id)
    assert {"p-aaa", "p-bbb"} <= excluded


@pytest.mark.asyncio
async def test_checkout_exclusion_falls_back_to_product_id():
    user_id = await _seed_user([
        {"event_type": EventType.CHECKOUT_START, "product_id": "p-solo",
         "event_metadata": {"count": 1}},
    ])
    excluded = await get_excluded_product_ids(user_id)
    assert "p-solo" in excluded


@pytest.mark.asyncio
async def test_add_to_cart_exclusion():
    user_id = await _seed_user([{"event_type": EventType.ADD_TO_CART, "product_id": "p-cart"}])
    assert "p-cart" in await get_excluded_product_ids(user_id)


@pytest.mark.asyncio
async def test_previous_recommendation_exclusion():
    user_id = await _seed_user([], recommendation_products=["p-rec-a", "p-rec-b"])
    excluded = await get_excluded_product_ids(user_id)
    assert {"p-rec-a", "p-rec-b"} <= excluded


@pytest.mark.asyncio
async def test_recent_view_exclusion():
    user_id = await _seed_user([
        {"event_type": EventType.PRODUCT_VIEW, "product_id": "p-viewed",
         "created_at": datetime.now(timezone.utc) - timedelta(hours=1)},
    ])
    assert "p-viewed" in await get_excluded_product_ids(user_id)


@pytest.mark.asyncio
async def test_old_view_is_not_excluded():
    """Views older than the 24h window must NOT exclude — recency matters."""
    user_id = await _seed_user([
        {"event_type": EventType.PRODUCT_VIEW, "product_id": "p-old",
         "created_at": datetime.now(timezone.utc) - timedelta(hours=RECENT_VIEW_WINDOW_HOURS + 1)},
    ])
    assert "p-old" not in await get_excluded_product_ids(user_id)


@pytest.mark.asyncio
async def test_exclusions_are_per_user():
    engaged = await _seed_user([
        {"event_type": EventType.ADD_TO_CART, "product_id": "p-x"},
        {"event_type": EventType.CHECKOUT_START, "event_metadata": {"product_ids": ["p-x"]}},
    ], recommendation_products=["p-x"])
    stranger = await _seed_user([])
    assert "p-x" in await get_excluded_product_ids(engaged)
    assert "p-x" not in await get_excluded_product_ids(stranger)


@pytest.mark.asyncio
async def test_no_activity_returns_empty_set():
    user_id = await _seed_user([])
    assert await get_excluded_product_ids(user_id) == set()