"""
One-off smoke test for the daily digest job logic — calls run_daily_digest()
directly rather than waiting for the actual cron schedule. This tests the
job's logic (find active users, run the agent for each), not the
scheduling mechanism itself (which is just APScheduler's own well-tested
cron matching — not worth re-testing here).
"""
import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
from app.services import llm_client, vector_store
from app.services.scheduler import run_daily_digest


async def main():
    async with AsyncSessionLocal() as db:
        print("Setting up: one product, two users (one active recently, one stale)...")

        product = Product(
            title="Agentic AI Fundamentals",
            description="Learn to build reasoning agents with LangGraph.",
            category="AI", price=49.99, vector_id=str(uuid.uuid4()),
        )
        db.add(product)
        await db.flush()
        embedding = await llm_client.get_embedding(product.to_embedding_text())
        await vector_store.upsert_product(
            vector_id=product.vector_id, embedding=embedding,
            document=product.to_embedding_text(),
            metadata={"category": product.category, "price": product.price, "sql_id": product.id},
        )

        active_user = User(email=f"active-{uuid.uuid4().hex[:6]}@example.com",
                            hashed_password=hash_password("test123"))
        stale_user = User(email=f"stale-{uuid.uuid4().hex[:6]}@example.com",
                           hashed_password=hash_password("test123"))
        db.add_all([active_user, stale_user])
        await db.flush()

        # Active user: event from right now (within the 24h lookback window)
        db.add(Event(user_id=active_user.id, event_type=EventType.SEARCH,
                      event_metadata={"query": "agentic ai"}))

        # Stale user: event manually backdated to 3 days ago (outside the window)
        stale_event = Event(user_id=stale_user.id, event_type=EventType.SEARCH,
                             event_metadata={"query": "old search"})
        stale_event.created_at = datetime.now(timezone.utc) - timedelta(days=3)
        db.add(stale_event)

        await db.commit()
        print(f"  Active user: {active_user.email}")
        print(f"  Stale user (3 days old activity): {stale_user.email}")

    print("\nRunning run_daily_digest()...\n")
    await run_daily_digest()

    print("\nChecking who got a recommendation...")
    async with AsyncSessionLocal() as db:
        active_rec = await db.execute(
            select(Recommendation).where(Recommendation.user_id == active_user.id)
        )
        stale_rec = await db.execute(
            select(Recommendation).where(Recommendation.user_id == stale_user.id)
        )
        active_has_rec = active_rec.scalar_one_or_none() is not None
        stale_has_rec = stale_rec.scalar_one_or_none() is not None

        print(f"  Active user got a recommendation: {active_has_rec} (expected: True)")
        print(f"  Stale user got a recommendation: {stale_has_rec} (expected: False)")

    print("\nCleaning up...")
    async with AsyncSessionLocal() as db:
        await vector_store.delete_product(product.vector_id)
        db_product = await db.get(Product, product.id)
        if db_product:
            await db.delete(db_product)
        for u in (active_user, stale_user):
            db_u = await db.get(User, u.id)
            if db_u:
                await db.delete(db_u)
        await db.commit()
    print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())