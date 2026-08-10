"""
retrieve node: builds a query from the user's cognitive model, embeds it,
and performs a metadata-filtered semantic search against the product
vector store. Returns candidates enriched with real Product data (not
just raw Chroma output), so downstream nodes don't need to touch the DB
again.
"""
from sqlalchemy import select

from app.agent.state import AgentState
from app.db.session import AsyncSessionLocal
from app.models.product import Product
from app.services import llm_client, vector_store

TOP_K = 10


def _build_query_text(cognitive_model: dict) -> str:
    """Turns the structured cognitive model into a single text query that
    captures the user's interest, for embedding."""
    parts = [
        cognitive_model.get("session_arc", ""),
        "Interests: " + ", ".join(cognitive_model.get("inferred_intents", [])),
        "Searched for: " + ", ".join(cognitive_model.get("stated_intents", [])),
    ]
    return " ".join(p for p in parts if p)


def _build_metadata_filter(cognitive_model: dict) -> dict | None:
    """Hybrid retrieval: narrow the semantic search to categories the user
    has actually shown affinity for, when we have that signal. Falls back
    to no filter (pure semantic search) if we don't yet know enough."""
    categories = cognitive_model.get("category_affinity", [])
    if not categories:
        return None
    if len(categories) == 1:
        return {"category": categories[0]}
    return {"category": {"$in": categories}}


async def retrieve_node(state: AgentState) -> dict:
    cognitive_model = state["cognitive_model"]

    query_text = _build_query_text(cognitive_model)
    query_embedding = await llm_client.get_embedding(query_text)

    where_filter = _build_metadata_filter(cognitive_model)
    raw_results = await vector_store.query(embedding=query_embedding, top_k=TOP_K, where=where_filter)

    # Chroma returns nested lists (one per query embedding); we only sent one.
    ids = raw_results.get("ids", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    sql_ids = [m.get("sql_id") for m in metadatas if m.get("sql_id")]
    if not sql_ids:
        return {"retrieved_candidates": []}

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Product).where(Product.id.in_(sql_ids)))
        products_by_id = {p.id: p for p in result.scalars().all()}

    candidates = []
    for vector_id, metadata, distance in zip(ids, metadatas, distances):
        sql_id = metadata.get("sql_id")
        product = products_by_id.get(sql_id)
        if not product:
            continue  # vector store had a stray entry not matched in SQL — skip, don't crash
        candidates.append({
            "product_id": product.id,
            "title": product.title,
            "description": product.description,
            "category": product.category,
            "price": product.price,
            "distance": distance,
        })

    return {"retrieved_candidates": candidates}