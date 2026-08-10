"""
One-off smoke test for the retrieve node in isolation. Creates two
products in different categories (dual-written to SQL + Chroma, same as
the real products router does), then checks that retrieve_node correctly
finds and filters candidates based on a synthetic cognitive model.
"""
import asyncio
import uuid

from sqlalchemy import select

from app.agent.nodes.retrieve import retrieve_node
from app.db.session import AsyncSessionLocal
from app.models.product import Product
from app.services import llm_client, vector_store


async def create_test_product(db, title: str, description: str, category: str, price: float) -> Product:
    product = Product(
        title=title, description=description, category=category, price=price,
        vector_id=str(uuid.uuid4()),
    )
    db.add(product)
    await db.flush()

    embedding = await llm_client.get_embedding(product.to_embedding_text())
    await vector_store.upsert_product(
        vector_id=product.vector_id,
        embedding=embedding,
        document=product.to_embedding_text(),
        metadata={"category": product.category, "price": product.price, "sql_id": product.id},
    )
    return product


async def main():
    async with AsyncSessionLocal() as db:
        ai_product = await create_test_product(
            db, "Agentic AI Fundamentals",
            "Learn to build reasoning agents with LangGraph and RAG pipelines.",
            "AI", 49.99,
        )
        baking_product = await create_test_product(
            db, "Introduction to Baking",
            "Learn to bake bread and pastries at home.",
            "Culinary", 19.99,
        )
        await db.commit()
        print(f"Created products: {ai_product.title} (AI), {baking_product.title} (Culinary)")

    fake_state = {
        "user_id": "test-user",
        "cognitive_model": {
            "session_arc": "User is exploring agentic AI and RAG systems.",
            "inferred_intents": ["agentic AI", "RAG pipelines"],
            "stated_intents": ["agentic ai"],
            "category_affinity": ["AI"],
        },
    }

    print("\nRunning retrieve_node with category_affinity=['AI']...")
    result = await retrieve_node(fake_state)

    print(f"\nRetrieved {len(result['retrieved_candidates'])} candidate(s):")
    for c in result["retrieved_candidates"]:
        print(f"  - {c['title']} (category: {c['category']}, distance: {c['distance']:.4f})")

    print("\nCleaning up test products...")
    async with AsyncSessionLocal() as db:
        for p in (ai_product, baking_product):
            await vector_store.delete_product(p.vector_id)
            db_product = await db.get(Product, p.id)
            if db_product:
                await db.delete(db_product)
        await db.commit()
    print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())