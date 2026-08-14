"""generate node — strategy selection, narrative generation, and the
grounding contract: the LLM may only contribute reasons, never product
metadata; duplicate/invented product IDs are dropped; output is never
empty."""
import json

import pytest

from app.agent.nodes.generate import generate_node, _enrich_products

CANDIDATES = [
    {"product_id": "c1", "title": "Agentic AI Fundamentals", "description": "LangGraph agents",
     "category": "AI", "price": 49.99, "relevance_score": 0.9},
    {"product_id": "c2", "title": "Production RAG", "description": "RAG at scale",
     "category": "AI", "price": 79.99, "relevance_score": 0.85},
    {"product_id": "c3", "title": "MLOps", "description": "ML in production",
     "category": "AI", "price": 89.99, "relevance_score": 0.8},
]


def make_state(**overrides):
    return {
        "cognitive_model": {
            "purchase_readiness": 0.3, "price_sensitivity": "medium",
            "decision_stage": "awareness", "detected_objections": [],
        },
        "filtered_candidates": CANDIDATES,
        **overrides,
    }


def test_strategy_selection_rules():
    cm = {"purchase_readiness": 0.65, "price_sensitivity": "medium",
          "decision_stage": "evaluation", "detected_objections": []}
    from app.agent.nodes.generate import _select_persuasion_strategy
    assert _select_persuasion_strategy(cm)[0] == "scarcity_urgency"
    assert _select_persuasion_strategy({**cm, "purchase_readiness": 0.3, "price_sensitivity": "high"})[0] == "social_proof"
    assert _select_persuasion_strategy({**cm, "purchase_readiness": 0.3, "price_sensitivity": "medium"})[0] == "authority_credibility"
    assert _select_persuasion_strategy({**cm, "purchase_readiness": 0.3, "decision_stage": "interest"})[0] == "curiosity_framing"


def test_enrich_uses_real_catalog_data_only():
    reasons = [{"product_id": "c1", "reason": "great fit"}, {"product_id": "hallucinated", "reason": "fake"}]
    enriched = _enrich_products(reasons, CANDIDATES)
    ids = [e["product_id"] for e in enriched]
    assert "hallucinated" not in ids  # invented id dropped
    assert "c1" in ids
    assert all(e["title"] and e["price"] for e in enriched)  # real catalog data, never LLM output


def test_enrich_dedupes_repeated_ids():
    reasons = [{"product_id": "c1", "reason": "a"}, {"product_id": "c1", "reason": "b"}, {"product_id": "c2", "reason": "c"}]
    enriched = _enrich_products(reasons, CANDIDATES)
    assert [e["product_id"] for e in enriched] == ["c1", "c2"]


def test_enrich_pads_to_minimum_two():
    enriched = _enrich_products([{"product_id": "c1", "reason": "only one"}], CANDIDATES)
    assert len(enriched) == 2
    assert enriched[1]["product_id"] in {"c2", "c3"}


@pytest.mark.asyncio
async def test_generate_returns_grounded_recommendation(fake_llm):
    result = await generate_node(make_state())
    assert result["persuasion_strategy"] == "curiosity_framing"
    assert result["narrative"]
    assert len(result["recommended_products"]) >= 2
    assert all(p["product_id"] in {"c1", "c2", "c3"} for p in result["recommended_products"])


@pytest.mark.asyncio
async def test_generate_degrades_gracefully_on_garbage_llm(fake_llm):
    fake_llm.responses = {"writing a personalized": "totally broken output"}
    result = await generate_node(make_state())
    # fallback: neutral narrative + real candidates as products
    assert result["narrative"]
    assert len(result["recommended_products"]) >= 2
    assert result["reasoning_chain"][0].startswith("fallback")


@pytest.mark.asyncio
async def test_generate_empty_candidates_no_llm_call(fake_llm):
    calls_before = fake_llm.chat_calls
    result = await generate_node(make_state(filtered_candidates=[]))
    assert fake_llm.chat_calls == calls_before  # no LLM call wasted
    assert result["recommended_products"] == []
    assert result["persuasion_strategy"] == "none"