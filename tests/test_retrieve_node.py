"""retrieve node — builds the embedding query, applies the metadata filter,
and joins vector results back to real SQL products (grounding)."""
import pytest

from app.agent.nodes import retrieve as retrieve_mod
from app.agent.nodes.retrieve import retrieve_node
from app.db.session import AsyncSessionLocal
from app.models.product import Product

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
