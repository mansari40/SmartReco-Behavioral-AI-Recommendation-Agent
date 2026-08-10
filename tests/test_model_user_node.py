"""
One-off smoke test for the model_user node in isolation. Creates a test
user with realistic events, runs the node, and prints the resulting
cognitive model — separate from the full graph so failures here are easy
to pinpoint before wiring it into the larger pipeline.
"""
import asyncio
import uuid

from sqlalchemy import select

from app.agent.nodes.model_user import model_user_node
from app.db.session import AsyncSessionLocal
from app.models.cognitive_model import UserCognitiveModel
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.user import User
from app.security import hash_password


async def setup_test_data() -> str:
    async with AsyncSessionLocal() as db:
        email = f"agent-test-{uuid.uuid4().hex[:8]}@example.com"
        user = User(email=email, hashed_password=hash_password("testpass123"))
        db.add(user)
        await db.flush()

        product = Product(
            title="Agentic AI Fundamentals",
            description="Learn to build reasoning agents with LangGraph.",
            category="AI",
            price=49.99,
            vector_id=str(uuid.uuid4()),
        )
        db.add(product)
        await db.flush()

        events = [
            Event(user_id=user.id, event_type=EventType.SEARCH,
                  event_metadata={"query": "agentic ai"}),
            Event(user_id=user.id, event_type=EventType.PRODUCT_VIEW,
                  product_id=product.id, event_metadata={"time_spent": 120}),
            Event(user_id=user.id, event_type=EventType.PRODUCT_VIEW,
                  product_id=product.id, event_metadata={"time_spent": 90}),
            Event(user_id=user.id, event_type=EventType.SEARCH,
                  event_metadata={"query": "langgraph tutorial"}),
        ]
        db.add_all(events)
        await db.commit()

        print(f"Created test user {email} (id={user.id}) with {len(events)} events")
        return user.id


async def main():
    user_id = await setup_test_data()

    print("\nRunning model_user_node...")
    result = await model_user_node({"user_id": user_id, "trigger_reason": "manual test"})

    print("\nReturned cognitive_model:")
    for k, v in result["cognitive_model"].items():
        print(f"  {k}: {v}")

    async with AsyncSessionLocal() as db:
        row_result = await db.execute(
            select(UserCognitiveModel).where(UserCognitiveModel.user_id == user_id)
        )
        row = row_result.scalar_one()
        print(f"\nPersisted to DB — last_event_count_at_update: {row.last_event_count_at_update}")


if __name__ == "__main__":
    asyncio.run(main())