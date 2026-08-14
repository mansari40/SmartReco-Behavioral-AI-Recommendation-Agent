"""Product CRUD + SQL/vector dual-write synchronization. Every create/update
must land in both SQL and the vector store; deletes must remove the vector
entry so deleted products are never retrievable again."""
import pytest

from tests.conftest import auth, fake_embedding

from app.db.session import AsyncSessionLocal
from app.models.product import Product
from app.services import vector_store
from sqlalchemy import select


async def _create(client, token, **overrides):
    payload = {
        "title": "Test Course", "description": "A test course description",
        "category": "AI", "price": 49.99, **overrides,
    }
    resp = await client.post("/api/products", json=payload, headers=auth(token))
    assert resp.status_code == 201
    return resp.json()


async def test_create_product_requires_admin(client, user_token):
    resp = await client.post("/api/products", json={
        "title": "X", "description": "Y", "category": "AI", "price": 1,
    }, headers=auth(user_token))
    assert resp.status_code == 403


async def test_create_dual_writes_to_sql_and_vector(client, admin_token, fake_llm):
    product = await _create(client, admin_token)
    assert product["sync_status"] == "synced"
    assert fake_llm.embedding_calls >= 1

    # SQL side
    async with AsyncSessionLocal() as db:
        row = (await db.execute(select(Product).where(Product.id == product["id"]))).scalar_one()
        assert row.title == "Test Course"

    # Vector side — retrievable by embedding (query with the same text that
    # was embedded as the document)
    doc_text = "Test Course. Category: AI. A test course description"
    results = await vector_store.query(embedding=fake_embedding(doc_text), top_k=50)
    docs = results.get("documents", [[]])[0]
    assert any("Test Course" in doc for doc in docs)


async def test_update_re_embeds_and_stays_synced(client, admin_token, fake_llm):
    product = await _create(client, admin_token, title="Old Title")
    calls_before = fake_llm.embedding_calls

    resp = await client.patch(f"/api/products/{product['id']}", json={"title": "New Title"}, headers=auth(admin_token))
    assert resp.status_code == 200
    assert resp.json()["sync_status"] == "synced"
    assert fake_llm.embedding_calls > calls_before


async def test_delete_removes_from_vector_store(client, admin_token):
    product = await _create(client, admin_token)
    async with AsyncSessionLocal() as db:
        row = (await db.execute(select(Product).where(Product.id == product["id"]))).scalar_one()
        vector_id = row.vector_id

    resp = await client.delete(f"/api/products/{product['id']}", headers=auth(admin_token))
    assert resp.status_code == 204

    # SQL row gone
    async with AsyncSessionLocal() as db:
        assert (await db.execute(select(Product).where(Product.id == product["id"]))).scalar_one_or_none() is None

    # vector entry gone — query by its own embedding must not return it
    results = await vector_store.query(embedding=fake_embedding("Test Course"), top_k=10)
    ids = results.get("ids", [[]])[0]
    assert vector_id not in ids


async def test_delete_requires_admin(client, user_token, admin_token):
    product = await _create(client, admin_token)
    resp = await client.delete(f"/api/products/{product['id']}", headers=auth(user_token))
    assert resp.status_code == 403


async def test_products_listable_without_auth(client):
    resp = await client.get("/api/products")
    assert resp.status_code == 200