"""retrieve node — builds the embedding query, applies the progressive
metadata filter (newest category -> recent categories -> unfiltered), joins
vector results back to real SQL products (grounding), and excludes
already-engaged products on every retrieval attempt. Also covers the
deterministic adaptive-retrieval gate (assess)."""
import uuid

import pytest

from app.agent.nodes import retrieve as retrieve_mod
from app.agent.nodes.retrieve import (
    MAX_RETRIEVAL_ATTEMPTS,
    retrieve_node,
    assess_retrieval_node,
    retry_retrieve_node,
)
from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
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
    product = Product(title=title, description=description, category=category, price=price,
                      vector_id=str(uuid.uuid4()))
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


def test_query_text_orders_recent_categories_newest_first():
    """The newest interest must lead the query text (recent_categories is
    stored chronological, oldest first) — a Data Engineering -> AI shift
    must surface AI first, not drown it in older history."""
    cm = {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering", "AI"]}
    query = retrieve_mod._build_query_text(cm)
    assert query.index("AI") < query.index("Data Engineering")


def test_metadata_filter_is_progressive_and_newest_first():
    cm = {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering"]}
    # attempt 1: the newest single category (recency has the strongest pull)
    assert retrieve_mod._build_metadata_filter(cm, attempt=1) == {"category": "Data Engineering"}
    # attempt 2: all recent categories
    assert retrieve_mod._build_metadata_filter(cm, attempt=2) == {"category": "Data Engineering"}

    cm2 = {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering", "AI"]}
    assert retrieve_mod._build_metadata_filter(cm2, attempt=1) == {"category": "AI"}  # newest wins
    assert retrieve_mod._build_metadata_filter(cm2, attempt=2) == {"category": {"$in": ["Data Engineering", "AI"]}}

    # attempt 3: unfiltered — final fallback only
    assert retrieve_mod._build_metadata_filter(cm2, attempt=3) is None

    # no recent categories -> accumulated affinity fallback, never a blind unfiltered jump
    assert retrieve_mod._build_metadata_filter({**COGNITIVE_MODEL, "recent_categories": []}) == {"category": "AI"}
    assert retrieve_mod._build_metadata_filter(
        {**COGNITIVE_MODEL, "recent_categories": [], "category_affinity": []}) is None


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
async def test_retry_relaxes_filter_progressively_and_reuses_embedding(fake_llm, monkeypatch):
    """The adaptive retry must progress newest category -> recent categories
    -> unfiltered, reusing the EXACT embedding from retrieve (zero extra
    embedding/chat calls) and never jumping straight to an unfiltered query."""
    await _seed_product("Agentic AI Fundamentals", "LangGraph and RAG agent building", "AI", 49.99)
    await _seed_product("Intro to Baking", "bread and pastries at home", "Culinary", 19.99)

    calls = []
    original_query = vector_store.query

    async def fake_query(embedding, top_k=10, where=None):
        calls.append({"embedding": embedding, "where": where})
        return await original_query(embedding, top_k=top_k, where=where)

    monkeypatch.setattr(vector_store, "query", fake_query)

    embedding = fake_embedding("Agentic AI Fundamentals. Category: AI. LangGraph and RAG agent building")
    state = {
        "query_embedding": embedding,
        "user_id": "x",
        "cognitive_model": {**COGNITIVE_MODEL, "recent_categories": ["AI"]},
        "retrieval_attempt": 1,
    }
    result = await retry_retrieve_node(state)

    assert len(calls) == 1
    # attempt 2 keeps the recency constraint (all recent categories), it does
    # NOT drop the filter immediately
    assert calls[0]["where"] == {"category": "AI"}
    assert calls[0]["embedding"] == embedding  # embedding reused, not recomputed
    assert result["retrieval_adjusted"] is True
    assert result["retrieval_attempt"] == 2
    assert result["retrieval_filter_applied"] is True

    # final stage (attempt 2 -> 3) relaxes to unfiltered — only as a fallback
    state["retrieval_attempt"] = 2
    result = await retry_retrieve_node(state)
    assert calls[-1]["where"] is None
    assert calls[-1]["embedding"] == embedding
    assert result["retrieval_attempt"] == MAX_RETRIEVAL_ATTEMPTS
    assert result["retrieval_filter_applied"] is False


async def _seed_user_with_events(events, recommendation_products=()):
    """User + events (+ optional prior recommendation) for exclusion tests."""
    async with AsyncSessionLocal() as db:
        user = User(email=f"excl-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
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
async def test_retrieve_excludes_engaged_products():
    de1 = await _seed_product("Data Quality Engineering", "quality validation and monitoring", "Data Engineering", 49.99)
    de2 = await _seed_product("Data Warehousing Fundamentals", "warehouse modeling and ETL", "Data Engineering", 44.99)
    fresh = await _seed_product("dbt Analytics Engineering", "dbt transformations and testing", "Data Engineering", 59.99)

    user_id = await _seed_user_with_events([
        {"event_type": EventType.PRODUCT_VIEW, "product_id": de1},
        {"event_type": EventType.ADD_TO_CART, "product_id": de1},
        {"event_type": EventType.CHECKOUT_START, "event_metadata": {"product_ids": [de1, de2]}},
    ], recommendation_products=[de2])

    cm = {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering"]}
    result = await retrieve_node({"user_id": user_id, "cognitive_model": cm})

    ids = {c["product_id"] for c in result["retrieved_candidates"]}
    assert fresh in ids  # genuinely unused course is still available
    assert de1 not in ids  # viewed + carted + checked out
    assert de2 not in ids  # checked out + previously recommended


@pytest.mark.asyncio
async def test_retry_keeps_exclusions_on_unfiltered_attempt(fake_llm, monkeypatch):
    """Exclusions must survive the relaxed/unfiltered retry — an engaged
    course can never slip back in through the final fallback query."""
    de1 = await _seed_product("Data Quality Engineering", "quality validation and monitoring", "Data Engineering", 49.99)
    fresh = await _seed_product("dbt Analytics Engineering", "dbt transformations and testing", "Data Engineering", 59.99)
    baking = await _seed_product("Intro to Baking", "bread and pastries at home", "Culinary", 19.99)

    user_id = await _seed_user_with_events([
        {"event_type": EventType.ADD_TO_CART, "product_id": de1},
        {"event_type": EventType.CHECKOUT_START, "event_metadata": {"product_ids": [de1]}},
    ])

    calls = []
    original_query = vector_store.query

    async def fake_query(embedding, top_k=10, where=None):
        calls.append(where)
        return await original_query(embedding, top_k=top_k, where=where)

    monkeypatch.setattr(vector_store, "query", fake_query)

    state = {
        "query_embedding": fake_embedding("data engineering courses"),
        "user_id": user_id,
        "cognitive_model": {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering"]},
        "retrieval_attempt": 1,
    }
    result = await retry_retrieve_node(state)
    assert result["retrieval_attempt"] == 2
    assert result["retrieval_filter_applied"] is True

    state["retrieval_attempt"] = 2
    result = await retry_retrieve_node(state)

    assert calls[-1] is None  # unfiltered fallback stage
    ids = {c["product_id"] for c in result["retrieved_candidates"]}
    assert de1 not in ids  # checked-out course stays excluded even unfiltered
    assert fresh in ids
    assert baking in ids  # unfiltered pool reaches other categories


@pytest.mark.asyncio
async def test_exclusions_do_not_leak_between_users():
    de1 = await _seed_product("Data Quality Engineering", "quality validation and monitoring", "Data Engineering", 49.99)

    other = await _seed_user_with_events([
        {"event_type": EventType.ADD_TO_CART, "product_id": de1},
        {"event_type": EventType.CHECKOUT_START, "event_metadata": {"product_ids": [de1]}},
    ])
    # a different user with NO activity must still see the course
    result = await retrieve_node({"user_id": "fresh-user", "cognitive_model": {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering"]}})
    ids = {c["product_id"] for c in result["retrieved_candidates"]}
    assert de1 in ids
    # ...while the engaged user must not
    result2 = await retrieve_node({"user_id": other, "cognitive_model": {**COGNITIVE_MODEL, "recent_categories": ["Data Engineering"]}})
    assert de1 not in {c["product_id"] for c in result2["retrieved_candidates"]}