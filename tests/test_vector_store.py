"""
One-off smoke test for the vector store service. Not part of the app —
safe to keep in tests/ or delete once confirmed working.

Uses get_embedding() from mesh_client, which respects MOCK_EMBEDDINGS in
.env — so this runs fine even with $0 Mesh balance, as long as
MOCK_EMBEDDINGS=true is set locally.
"""
import asyncio

from app.services import llm_client, vector_store

async def main():
    print("Generating embeddings for two sample products...")
    text_a = "Agentic AI Fundamentals. Category: AI. Learn to build reasoning agents with LangGraph."
    text_b = "Introduction to Baking. Category: Culinary. Learn to bake bread and pastries at home."

    emb_a = await llm_client.get_embedding(text_a)
    emb_b = await llm_client.get_embedding(text_b)
    print(f"Embedding A length: {len(emb_a)}")
    print(f"Embedding B length: {len(emb_b)}")

    print("\nUpserting both into Chroma...")
    await vector_store.upsert_product(
        vector_id="test-product-a",
        embedding=emb_a,
        document=text_a,
        metadata={"category": "AI", "price": 49.99},
    )
    await vector_store.upsert_product(
        vector_id="test-product-b",
        embedding=emb_b,
        document=text_b,
        metadata={"category": "Culinary", "price": 19.99},
    )
    print("Upserts complete.")

    print("\nQuerying with product A's own embedding (should return A as closest match)...")
    results = await vector_store.query(embedding=emb_a, top_k=2)
    print("Query results:", results["ids"], results["documents"])

    print("\nQuerying with a metadata filter (category=AI only)...")
    filtered = await vector_store.query(embedding=emb_a, top_k=2, where={"category": "AI"})
    print("Filtered results:", filtered["ids"])

    print("\nDeleting both test products...")
    await vector_store.delete_product("test-product-a")
    await vector_store.delete_product("test-product-b")
    print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())