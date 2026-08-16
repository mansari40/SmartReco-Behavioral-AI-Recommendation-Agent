"""Full agent graph end-to-end (offline, mocked LLM): behavior → cognitive
model → retrieval → evaluation → filtering → generation → persistence.
Verifies the stored recommendation is grounded in the real catalog and that
engagement exclusions + the relevance floor shape the content.

Query text is pinned to a known catalog product's embedding text so the
semantic self-match (distance 0 -> similarity 1) is deterministic — the
relevance floor is never lowered or disabled for tests."""
import uuid

import pytest

from app.agent.graph import agent_graph
from app.db.session import AsyncSessionLocal
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.security import hash_password
from app.services import vector_store
from sqlalchemy import select
from tests.conftest import fake_embedding

RAG_TEXT = "Production RAG at Scale. Category: AI. Advanced retrieval-augmented generation patterns."
BAKING_TEXT = "Introduction to Baking. Category: Culinary. Bread and pastries at home."


async def _seed_catalog_and_user():
    async with AsyncSessionLocal() as db:
        products = []
        for title, desc, cat, price in [
            ("Agentic AI Fundamentals", "Build reasoning agents with LangGraph and RAG.", "AI", 49.99),
            ("Production RAG at Scale", "Advanced retrieval-augmented generation patterns.", "AI", 79.99),
            ("Introduction to Baking", "Bread and pastries at home.", "Culinary", 19.99),
        ]:
            p = Product(title=title, description=desc, category=cat, price=price, vector_id=str(uuid.uuid4()))
            db.add(p)
            await db.flush()
            await vector_store.upsert_product(
                vector_id=p.vector_id, embedding=fake_embedding(p.to_embedding_text()),
                document=p.to_embedding_text(),
                metadata={"category": p.category, "price": p.price, "sql_id": p.id},
            )
            products.append(p)

        user = User(email=f"e2e-{uuid.uuid4().hex[:8]}@test.com", hashed_password=hash_password("x"))
        db.add(user)
        await db.flush()
        db.add_all([
            Event(user_id=user.id, event_type=EventType.SEARCH, event_metadata={"query": "agentic ai"}),
            Event(user_id=user.id, event_type=EventType.PRODUCT_VIEW, product_id=products[0].id,
                  event_metadata={"title": products[0].title, "category": "AI"}),
            Event(user_id=user.id, event_type=EventType.SEARCH, event_metadata={"query": "langgraph rag"}),
        ])
        await db.commit()
        return user.id


async def _active_products(user_id):
    async with AsyncSessionLocal() as db:
        rec = (await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )).scalar_one_or_none()
        if rec is None:
            return None
        return {p["product_id"] for p in rec.products}, rec


@pytest.mark.asyncio
async def test_full_graph_runs_and_persists_grounded_recommendation(fake_llm, monkeypatch):
    user_id = await _seed_catalog_and_user()

    from app.agent.nodes import retrieve as retrieve_mod

    # Pin the query to RAG's embedding text: deterministic self-match (sim 1.0)
    # clears the relevance floor; Baking scores ~0.11 vs this query (< floor).
    monkeypatch.setattr(retrieve_mod, "_build_query_text", lambda cm: RAG_TEXT)

    final_state = await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "pytest e2e"})

    # every pipeline stage ran
    assert final_state.get("cognitive_model")
    assert final_state.get("narrative")
    assert final_state.get("confidence") is not None
    assert final_state.get("persuasion_strategy")

    # the viewed course is excluded; the valid unused course is recommended
    async with AsyncSessionLocal() as db:
        stored = (await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )).scalar_one_or_none()
        assert stored is not None
        assert stored.narrative == final_state["narrative"]
        assert stored.trigger_reason == "pytest e2e"

        products = {p["product_id"]: p for p in stored.products}
        db_products = (await db.execute(select(Product))).scalars().all()
        by_title = {p.title: p for p in db_products}
        assert by_title["Production RAG at Scale"].id in products  # genuine match, above floor
        assert by_title["Agentic AI Fundamentals"].id not in products  # viewed -> excluded
        # near-floor candidates must not be force-filled: the only stored
        # product is the one that genuinely cleared the relevance floor
        assert len(products) >= 1
        for p in stored.products:
            assert p["score"] >= 0.2  # nothing below the relevance floor

    # the agent's normal path makes exactly 3 AI calls: 2 chat (model_user,
    # generate) + 1 embedding (retrieve). Adaptive retries reuse the same
    # embedding and the deterministic nodes make no calls at all.
    assert fake_llm.chat_calls == 2
    assert fake_llm.embedding_calls == 1
    assert fake_llm.chat_calls + fake_llm.embedding_calls == 3


@pytest.mark.asyncio
async def test_agent_is_idempotent_across_runs(fake_llm, monkeypatch):
    """Re-running the agent must deactivate the previous recommendation and
    never duplicate the active one — and a NEW run with a genuinely new
    unused candidate (a shift in interest) replaces the old set."""
    user_id = await _seed_catalog_and_user()

    from app.agent.nodes import retrieve as retrieve_mod

    # run 1 pins the query to the AI course; run 2 to the culinary course —
    # deterministic self-matches that exercise the shift in interest.
    queries = iter([RAG_TEXT, BAKING_TEXT])
    monkeypatch.setattr(retrieve_mod, "_build_query_text", lambda cm: next(queries))

    await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "run 1"})
    await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "run 2"})

    async with AsyncSessionLocal() as db:
        active = (await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )).scalars().all()
        assert len(active) == 1
        assert active[0].trigger_reason == "run 2"

        baking = (await db.execute(select(Product).where(Product.title == "Introduction to Baking"))).scalar_one()
        assert baking.id in {p["product_id"] for p in active[0].products}  # run 2 reflects the new interest

        history = (await db.execute(select(Recommendation).where(Recommendation.user_id == user_id))).scalars().all()
        assert len(history) == 2  # history preserved, only latest active


@pytest.mark.asyncio
async def test_adaptive_retrieval_retries_on_poor_quality(fake_llm, monkeypatch):
    """Poor retrieval quality (weak best similarity) with an applied category
    filter must route through the deterministic progressive relax retry — and
    when nothing clears the relevance floor, the run must NOT force a
    recommendation (no generation call, no stored rec)."""
    user_id = await _seed_catalog_and_user()

    from app.agent.nodes import retrieve as retrieve_mod
    from app.agent.nodes import evaluate as evaluate_mod

    monkeypatch.setattr(retrieve_mod, "_best_similarity", lambda candidates: 0.0)
    # force every candidate below the floor: no generation may happen
    monkeypatch.setattr(evaluate_mod, "_similarity", lambda candidate: 0.0)

    final_state = await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "poor quality"})

    assert final_state.get("retrieval_quality") == "low"
    assert final_state.get("retrieval_adjusted") is True
    assert final_state.get("filtered_candidates") == []
    # the learning fallback narrative — and NO generation call, NO stored rec
    assert final_state.get("narrative")
    assert fake_llm.chat_calls == 1  # model_user only — nothing was forced
    assert fake_llm.embedding_calls == 1
    async with AsyncSessionLocal() as db:
        rec = (await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )).scalar_one_or_none()
        assert rec is None  # store guard: no empty recommendation


@pytest.mark.asyncio
async def test_adaptive_retrieval_skips_retry_on_good_quality(fake_llm, monkeypatch):
    """Good retrieval quality at the FIRST attempt flows straight through —
    no retry, no adjustment."""
    user_id = await _seed_catalog_and_user()

    # one extra unused AI course so the first (category-filtered) attempt has
    # >= MIN_RETRIEVAL_RESULTS candidates even after the viewed course is
    # excluded — good quality at attempt 1, so no retry should occur
    async with AsyncSessionLocal() as db:
        p = Product(title="LangGraph Systems", description="System-level agent orchestration.",
                    category="AI", price=69.99, vector_id=str(uuid.uuid4()))
        db.add(p)
        await db.flush()
        await vector_store.upsert_product(
            vector_id=p.vector_id, embedding=fake_embedding(p.to_embedding_text()),
            document=p.to_embedding_text(),
            metadata={"category": p.category, "price": p.price, "sql_id": p.id},
        )
        await db.commit()

    from app.agent.nodes import retrieve as retrieve_mod
    from app.agent.nodes import evaluate as evaluate_mod

    monkeypatch.setattr(retrieve_mod, "_best_similarity", lambda candidates: 0.9)
    monkeypatch.setattr(evaluate_mod, "_similarity", lambda candidate: 0.9)  # all above floor

    final_state = await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "good quality"})

    assert final_state.get("retrieval_quality") == "good"
    assert final_state.get("retrieval_adjusted") is not True
    assert fake_llm.chat_calls == 2
    assert fake_llm.embedding_calls == 1


@pytest.mark.asyncio
async def test_agent_keeps_good_recommendation_when_new_run_has_no_valid_candidates(fake_llm, monkeypatch):
    """The stale-recommendation guard: when a new run produces zero valid
    candidates (everything engaged + nothing above the floor), the previous
    active recommendation must survive untouched."""
    user_id = await _seed_catalog_and_user()

    from app.agent.nodes import retrieve as retrieve_mod

    # run 1: pin the query to RAG (deterministic match, valid rec)
    monkeypatch.setattr(retrieve_mod, "_build_query_text", lambda cm: RAG_TEXT)
    await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "run 1"})

    # run 2: force every candidate to score ~0 — nothing clears the floor
    from app.agent.nodes import evaluate as evaluate_mod
    monkeypatch.setattr(evaluate_mod, "_similarity", lambda candidate: 0.0)
    await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": "run 2 empty"})

    async with AsyncSessionLocal() as db:
        active = (await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id, Recommendation.is_active.is_(True)
            )
        )).scalars().all()
        assert len(active) == 1
        assert active[0].trigger_reason == "run 1"  # the good old rec survived
        assert len(active[0].products) > 0