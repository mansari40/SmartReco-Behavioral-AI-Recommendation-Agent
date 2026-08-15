"""retrieve node — builds the embedding query, applies the metadata filter,
and joins vector results back to real SQL products (grounding). Also covers
the deterministic adaptive-retrieval gate (assess) and the filter-relaxing
retry that reuses the already-computed embedding."""
import pytest

from app.agent.nodes import retrieve as retrieve_mod
from app.agent.nodes.retrieve import (
    retrieve_node,
    assess_retrieval_node,
    retry_retrieve_node,
)
from app.db.session import AsyncSessionLocal
from app.models.product import Product
from app.services import vector_store
from tests.conftest import fake_embedding

COGNITIVE_MODEL = {
    "session_arc": "User explored agentic AI and LangGraph.",
    "inferred_intents": ["building AI agents"],
    "stated_intents": ["agentic ai"],
    "category_affinity": ["AI"],
    "recent_searches": [],
    "recent_categories": [],
}


async def _seed_product(title, description, category, price):
    from app.services import llm_client, vector_store
    import uuid as _uuid
    from tests.conftest import fake_embedding

    product = Product(title=title, description=description, category=category, price=price,
                      vector_id=str(_uuid.uuid4()))
    async with AsyncSessionLocal() as db:
        db.add(product)
        await db.flush()
        await vector_store.upsert_product(
            vector_id=product.vector_id,
            embedding=fake_embedding(product.to_embedding_text()),
            document=product.to_embedding_text(),
            metadata={"category": product.category, "price": product.price, "sql_id": product.id},
        )
        await db.commit()
        return product.id


def test_query_text_leads_with_recent_signals():
    cm = {
        "recent_searches": ["Kafka", "dbt"],
        "recent_categories": ["Data Engineering"],
        "session_arc": "arc",
        "inferred_intents": ["data"],
        "stated_intents": ["old interest"],
    }
    query = retrieve_mod._build_query_text(cm)
    assert query.index("Kafka") < query.index("old interest")  # recency leads
    assert "Data Engineering" in query


def test_metadata_filter_uses_recent_categories_first():
    cm = {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering"], "category_affinity": ["AI"]}
    assert retrieve_mod._build_metadata_filter(cm) == {"category": "Data Engineering"}  # single category → scalar

    cm2 = {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering", "Data Science"], "category_affinity": ["AI"]}
    assert retrieve_mod._build_metadata_filter(cm2) == {"category": {"$in": ["Data Engineering", "Data Science"]}}

    assert retrieve_mod._build_metadata_filter({**COGNITIVE_MODEL, "recent_categories": []}) == {"category": "AI"}
    assert retrieve_mod._build_metadata_filter({**COGNITIVE_MODEL, "recent_categories": [], "category_affinity": []}) is None


@pytest.mark.asyncio
async def test_retrieve_returns_sql_grounded_candidates():
    ai_id = await _seed_product("Agentic AI Fundamentals", "LangGraph and RAG agent building", "AI", 49.99)
    baking_id = await _seed_product("Intro to Baking", "bread and pastries at home", "Culinary", 19.99)

    result = await retrieve_node({"user_id": "x", "cognitive_model": {**COGNITIVE_MODEL, "recent_categories": ["AI"]}})
    titles = [c["title"] for c in result["retrieved_candidates"]]
    assert "Agentic AI Fundamentals" in titles
    assert "Intro to Baking" not in titles  # metadata filter excludes Culinary

    candidates = {c["product_id"]: c for c in result["retrieved_candidates"]}
    assert ai_id in candidates
    assert candidates[ai_id]["price"] == 49.99  # real catalog data, not LLM output


def test_query_text_falls_back_to_neutral_query():
    assert retrieve_mod._build_query_text({}) == ""
    assert retrieve_mod.DEFAULT_QUERY_TEXT  # retrieve_node must never embed an empty string


def test_quality_gate_is_deterministic():
    assert retrieve_mod._best_similarity([]) == 0.0
    assert retrieve_mod._best_similarity([{"distance": 0.1}, {"distance": 0.5}]) == 0.9
    assert retrieve_mod._best_similarity([{"distance": 1.5}]) == 0.0  # clamped to [0, 1]
    assert retrieve_mod._quality([{"distance": 0.1}, {"distance": 0.2}]) == "good"
    assert retrieve_mod._quality([{"distance": 1.5}, {"distance": 0.9}]) == "low"
    assert retrieve_mod._quality([]) == "low"


@pytest.mark.asyncio
async def test_assess_marks_quality_without_llm():
    assert (await assess_retrieval_node(
        {"retrieved_candidates": [{"distance": 0.1}, {"distance": 0.2}]}
    ))["retrieval_quality"] == "good"
    assert (await assess_retrieval_node({"retrieved_candidates": []}))["retrieval_quality"] == "low"
    assert (await assess_retrieval_node(
        {"retrieved_candidates": [{"distance": 1.2}]}
    ))["retrieval_quality"] == "low"


@pytest.mark.asyncio
async def test_retry_relaxes_filter_and_reuses_embedding(fake_llm, monkeypatch):
    """The adaptive retry must re-query with NO filter and the EXACT same
    embedding retrieved already — a deterministic adjustment with zero extra
    embedding/chat calls."""
    await _seed_product("Agentic AI Fundamentals", "LangGraph and RAG agent building", "AI", 49.99)
    await _seed_product("Intro to Baking", "bread and pastries at home", "Culinary", 19.99)

    calls = []
    original_query = vector_store.query

    async def fake_query(embedding, top_k=10, where=None):
        calls.append({"embedding": embedding, "where": where})
        return await original_query(embedding, top_k=top_k, where=where)

    monkeypatch.setattr(vector_store, "query", fake_query)

    # Query with a real product's own embedding: the self-match (distance 0)
    # is guaranteed to rank first, so grounding is deterministic even with
    # orphaned vectors from earlier tests in the shared Chroma collection.
    embedding = fake_embedding("Agentic AI Fundamentals. Category: AI. LangGraph and RAG agent building")
    result = await retry_retrieve_node({"query_embedding": embedding})

    assert len(calls) == 1
    assert calls[0]["where"] is None  # filter relaxed
    assert calls[0]["embedding"] == embedding  # embedding reused, not recomputed
    assert result["retrieval_adjusted"] is True
    titles = [c["title"] for c in result["retrieved_candidates"]]
    assert "Agentic AI Fundamentals" in titles
    assert "Intro to Baking" in titles  # no filter -> both categories are reachable
