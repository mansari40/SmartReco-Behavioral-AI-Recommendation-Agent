"""
retrieve node: builds a query from the user's cognitive model, embeds it,
and performs a metadata-filtered semantic search against the product
vector store. Returns candidates enriched with real Product data (not
just raw Chroma output), so downstream nodes don't need to touch the DB
again.

The retrieval pipeline is conditionally adaptive without any extra LLM
spend:

    retrieve -> assess -> (retry -> evaluate  |  evaluate)

assess deterministically scores the quality of what came back (candidate
count + best similarity). If quality is poor AND a category filter was
narrowing the search, the graph routes to retry, which relaxes the filter
and re-queries with the SAME embedding already computed by retrieve — a
deterministic adjustment that costs zero additional AI calls.
"""
from sqlalchemy import select

from app.agent.state import AgentState
from app.db.session import AsyncSessionLocal
from app.models.product import Product
from app.services import llm_client, vector_store
from app.services.engagement_filter import get_excluded_product_ids

TOP_K = 10
# The vector store is queried with a larger pool than we actually keep, and
# the result is grounded against SQL and capped at TOP_K. This stops stale
# vectors (products whose SQL row is gone after a delete, or a failed
# dual-write) from crowding valid current products out of the top-K before
# SQL filtering — observed with as few as ~40 orphans.
RETRIEVAL_POOL_MARGIN = 3
# Deterministic retrieval-quality gate. A search is "good" when it returns at
# least MIN_RETRIEVAL_RESULTS candidates whose best cosine similarity is at
# least MIN_RETRIEVAL_SIMILARITY (distance <= 0.8). Anything below that means
# the metadata filter likely over-constrained the search — worth a relax retry.
MIN_RETRIEVAL_RESULTS = 2
MIN_RETRIEVAL_SIMILARITY = 0.2
# Used only when the user has no retrievable signals at all — a neutral query
# so semantic search still returns grounded catalog candidates instead of an
# empty embedding.
DEFAULT_QUERY_TEXT = "interesting online courses"


def _build_query_text(cognitive_model: dict) -> str:
    """Turns the structured cognitive model into a single text query for
    embedding. Recent behavior leads the query — what the user is doing
    RIGHT NOW matters more than older accumulated interests — while older
    stated intents still contribute so the query isn't blind to history."""
    parts = []

    recent_searches = cognitive_model.get("recent_searches", [])
    if recent_searches:
        parts.append("Recently searching for: " + ", ".join(recent_searches))

    recent_categories = cognitive_model.get("recent_categories", [])
    if recent_categories:
        # Newest interest leads the query (recent_categories is stored
        # chronological, oldest first) so the embedding is dominated by what
        # the user is engaging with RIGHT NOW, not their older interests.
        parts.append("Recently viewing courses in: " + ", ".join(reversed(recent_categories)))

    arc = cognitive_model.get("session_arc", "")
    if arc:
        parts.append(arc)

    inferred = cognitive_model.get("inferred_intents", [])
    if inferred:
        parts.append("Interests: " + ", ".join(inferred))

    stated = cognitive_model.get("stated_intents", [])
    if stated:
        parts.append("Also searched for: " + ", ".join(stated))

    return " ".join(p for p in parts if p)


MAX_RETRIEVAL_ATTEMPTS = 3


def _build_metadata_filter(cognitive_model: dict, attempt: int = 1) -> dict | None:
    """Progressive recency-preserving metadata filter.

    Attempt 1: the user's NEWEST category alone — the newest interest gets
              the strongest influence.
    Attempt 2: all recent categories ($in) — broader, still recency-bound.
    Attempt 3: no filter — pure semantic search, final fallback only.

    Falls back to accumulated category affinity (all attempts) for users
    with no recent category signals. The user's historical dominant
    interest must never override the newest one (DE must not drown out AI)."""
    recent_categories = cognitive_model.get("recent_categories", [])

    if attempt >= MAX_RETRIEVAL_ATTEMPTS:
        return None  # final stage is always unfiltered

    if attempt == 1 and recent_categories:
        return {"category": recent_categories[-1]}

    if attempt == 2 and recent_categories:
        if len(recent_categories) == 1:
            return {"category": recent_categories[0]}
        return {"category": {"$in": recent_categories}}

    categories = cognitive_model.get("category_affinity", [])
    if not categories:
        return None
    if len(categories) == 1:
        return {"category": categories[0]}
    return {"category": {"$in": categories}}


async def _extract_candidates(raw_results: dict, limit: int = TOP_K) -> list[dict]:
    """Normalize Chroma/pgvector query output and join vector hits back to
    real SQL Product rows (grounding). Stray vector entries with no matching
    SQL row are skipped, never fatal. The raw query used a larger pool than
    `limit` so orphaned vectors can't crowd out valid products."""
    ids = raw_results.get("ids", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    sql_ids = [m.get("sql_id") for m in metadatas if m.get("sql_id")]
    if not sql_ids:
        return []

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
            "level": product.level,
            "rating": product.rating,
            "rating_count": product.rating_count,
            "distance": distance,
        })

    return candidates[:limit]


async def _apply_exclusions(candidates: list[dict], user_id: str) -> list[dict]:
    """Drop products the user has already engaged with — checked out, added
    to cart, previously recommended, or viewed within the last 24 hours —
    from the candidate pool. Applied on EVERY retrieval attempt (normal
    filtered query and every retry), so re-served engaged courses can never
    slip back in through the relaxed/unfiltered path."""
    if not candidates:
        return []
    excluded = await get_excluded_product_ids(user_id)
    if not excluded:
        return candidates
    return [c for c in candidates if c["product_id"] not in excluded]


def _best_similarity(candidates: list[dict]) -> float:
    """Best cosine similarity across the candidates (1 - distance, clamped to
    [0, 1]). Candidates without a distance count as neutral 0.5, matching the
    evaluate node's convention. Used only as a deterministic quality signal."""
    similarities = []
    for c in candidates:
        distance = c.get("distance")
        if distance is None:
            similarities.append(0.5)
            continue
        try:
            similarities.append(max(0.0, min(1.0, 1.0 - float(distance))))
        except (TypeError, ValueError):
            similarities.append(0.5)
    return max(similarities) if similarities else 0.0


def _quality(candidates: list[dict]) -> str:
    return (
        "good"
        if len(candidates) >= MIN_RETRIEVAL_RESULTS
        and _best_similarity(candidates) >= MIN_RETRIEVAL_SIMILARITY
        else "low"
    )


async def retrieve_node(state: AgentState) -> dict:
    cognitive_model = state["cognitive_model"]
    user_id = state["user_id"]

    query_text = _build_query_text(cognitive_model) or DEFAULT_QUERY_TEXT
    query_embedding = await llm_client.get_embedding(query_text)

    where_filter = _build_metadata_filter(cognitive_model, attempt=1)
    raw_results = await vector_store.query(
        embedding=query_embedding, top_k=TOP_K * RETRIEVAL_POOL_MARGIN, where=where_filter
    )

    candidates = await _extract_candidates(raw_results)
    candidates = await _apply_exclusions(candidates, user_id)

    return {
        "retrieved_candidates": candidates,
        "query_embedding": query_embedding,
        "retrieval_filter_applied": where_filter is not None,
        "retrieval_attempt": 1,
    }


async def assess_retrieval_node(state: AgentState) -> dict:
    """Deterministic retrieval-quality decision (no LLM). Marks the retrieval
    as good/low so the graph can route to evaluate directly or to the relax
    retry."""
    return {"retrieval_quality": _quality(state.get("retrieved_candidates", []))}


async def retry_retrieve_node(state: AgentState) -> dict:
    """Deterministic retrieval adjustment — progressive filter relaxation,
    never a blind jump to unfiltered. attempt 1 -> 2 -> 3 (newest category
    -> all recent categories -> unfiltered only as a final fallback), always
    reusing the exact embedding already computed by retrieve (zero extra
    embedding/chat calls) and re-applying the engaged-product exclusions."""
    query_embedding = state["query_embedding"]
    current_attempt = state.get("retrieval_attempt", 1)
    next_attempt = min(current_attempt + 1, MAX_RETRIEVAL_ATTEMPTS)

    where_filter = _build_metadata_filter(state.get("cognitive_model", {}), attempt=next_attempt)
    raw_results = await vector_store.query(
        embedding=query_embedding, top_k=TOP_K * RETRIEVAL_POOL_MARGIN, where=where_filter
    )
    candidates = await _extract_candidates(raw_results)
    candidates = await _apply_exclusions(candidates, state["user_id"])

    return {
        "retrieved_candidates": candidates,
        "retrieval_adjusted": True,
        "retrieval_attempt": next_attempt,
        "retrieval_filter_applied": where_filter is not None,
        "retrieval_quality": _quality(candidates),
    }