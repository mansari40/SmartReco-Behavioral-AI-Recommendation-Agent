"""Vector store service tests: upsert, semantic query, metadata filtering,
and delete (dual-write correctness lives in test_products.py)."""
import pytest

from app.services import vector_store
from tests.conftest import fake_embedding


@pytest.mark.asyncio
async def test_upsert_query_filter_delete_cycle():
    vid = "vs-test-product-1"
    text = "A unique course about deep sea biology and marine life."
    await vector_store.upsert_product(
        vector_id=vid, embedding=fake_embedding(text), document=text,
        metadata={"category": "Biology", "price": 25.0, "sql_id": "p-1"},
    )

    # self-query returns the product first
    results = await vector_store.query(embedding=fake_embedding(text), top_k=3)
    ids = results.get("ids", [[]])[0]
    assert vid in ids

    # metadata filter narrows results
    filtered = await vector_store.query(
        embedding=fake_embedding(text), top_k=3, where={"category": "Biology"}
    )
    assert vid in filtered.get("ids", [[]])[0]
    filtered_other = await vector_store.query(
        embedding=fake_embedding(text), top_k=3, where={"category": "Cooking"}
    )
    assert vid not in filtered_other.get("ids", [[]])[0]

    # delete removes it from retrieval entirely
    await vector_store.delete_product(vid)
    after = await vector_store.query(embedding=fake_embedding(text), top_k=5)
    assert vid not in after.get("ids", [[]])[0]