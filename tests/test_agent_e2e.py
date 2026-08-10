"""
Full end-to-end smoke test for the recommendation agent. Creates a
product catalog, a user with realistic events, runs the complete
LangGraph pipeline, and verifies the final Recommendation row actually
landed in the database correctly. This is the test that proves the whole
system works together, not just each piece in isolation.
"""
import asyncio
import uuid

from sqlalchemy import select

from app.agent.graph import agent_graph
from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
from app.services import llm_client, vector_store


async def seed_catalog(db) -> list[Product]:
    catalog = [
        ("Agentic AI Fundamentals", "Learn to build reasoning agents with LangGraph and RAG pipelines.", "AI", 49.99),
        ("Production RAG at Scale", "Advanced retrieval-augmented generation patterns for real-world systems.", "AI", 79.99),
        ("MLOps with Kubernetes", "Deploy and scale ML systems in production using Kubernetes.", "AI", 89.99),
        ("Introduction to Baking", "Learn to bake bread and pastries at home.", "Culinary", 19.99),
        ("Watercolor Painting Basics", "Fundamentals of watercolor technique for beginners.", "Art", 24.99),
    ]
    products = []
    for title, description, category, price in catalog:
        product = Product(title=title, description=description, category=category, price=price,
                           vector_id=str(uuid.uuid4()))
        db.add(product)
        await db.flush()
        embedding = await llm_client.get_embedding(product.to_embedding_text())
        await vector_store.upsert_product(
            vector_id=product.vector_id, embedding=embedding,
            document=product.to_embedding_text(),
            metadata={"category": product.category, "price": product.price, "sql_id": product.id},
        )
        products.append(product)
    await db.commit()
    return products


async def main():
    async with AsyncSessionLocal() as db:
        print("Seeding product catalog...")
        products = await seed_catalog(db)
        ai_product = next(p for p in products if p.category == "AI")
        print(f"  Created {len(products)} products across AI, Culinary, Art")

        print("\nCreating test user with behavioral events...")
        email = f"e2e-test-{uuid.uuid4().hex[:8]}@example.com"
        user = User(email=email, hashed_password=hash_password("testpass123"))
        db.add(user)
        await db.flush()

        events = [
            Event(user_id=user.id, event_type=EventType.SEARCH, event_metadata={"query": "agentic ai"}),
            Event(user_id=user.id, event_type=EventType.PRODUCT_VIEW, product_id=ai_product.id,
                  event_metadata={"time_spent": 150}),
            Event(user_id=user.id, event_type=EventType.SEARCH, event_metadata={"query": "langgraph rag"}),
            Event(user_id=user.id, event_type=EventType.PRODUCT_VIEW, product_id=ai_product.id,
                  event_metadata={"time_spent": 200}),
        ]
        db.add_all(events)
        await db.commit()
        print(f"  Created user {email} with {len(events)} events")
        user_id = user.id

    print("\nRunning full agent graph...\n")
    final_state = await agent_graph.ainvoke({
        "user_id": user_id,
        "trigger_reason": "e2e test: manual invocation",
    })

    print("=" * 60)
    print("FINAL STATE SUMMARY")
    print("=" * 60)
    print(f"Persuasion strategy: {final_state.get('persuasion_strategy')}")
    print(f"Confidence: {final_state.get('confidence')}")
    print(f"Regenerate count: {final_state.get('regenerate_count', 0)}")
    print(f"\nNarrative:\n  {final_state.get('narrative')}")
    print(f"\nRecommended products:")
    for p in final_state.get("recommended_products", []):
        print(f"  - {p}")

    print("\n" + "=" * 60)
    print("VERIFYING DATABASE PERSISTENCE")
    print("=" * 60)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )
        stored = result.scalar_one_or_none()
        if stored:
            print(f"Found active Recommendation row: id={stored.id}")
            print(f"  narrative matches state: {stored.narrative == final_state.get('narrative')}")
            print(f"  trigger_reason: {stored.trigger_reason}")
        else:
            print("ERROR: No active Recommendation row found in database!")

    print("\nCleaning up test data...")
    async with AsyncSessionLocal() as db:
        for p in products:
            await vector_store.delete_product(p.vector_id)
            db_p = await db.get(Product, p.id)
            if db_p:
                await db.delete(db_p)
        db_user = await db.get(User, user_id)
        if db_user:
            await db.delete(db_user)  # cascades to events, cognitive_model, recommendations
        await db.commit()
    print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())