"""evaluate node — deterministic relevance scoring against the cognitive
model (no LLM). Scores blend semantic distance with category affinity and
inferred-intent overlap; the output contract is unchanged (relevance_score +
evaluation_reasoning, sorted descending)."""
import pytest

from app.agent.nodes.evaluate import evaluate_node


def make_candidate(product_id, title="T", category="AI", description="d", distance=None):
    c = {
        "product_id": product_id,
        "title": title,
        "category": category,
        "description": description,
    }
    if distance is not None:
        c["distance"] = distance
    return c


def make_state(**overrides):
    return {
        "cognitive_model": {
            "inferred_intents": ["AI agents"],
            "category_affinity": ["AI"],
            "recent_categories": [],
        },
        "retrieved_candidates": [
            make_candidate("a1", title="Agentic AI", description="Build AI agents", category="AI", distance=0.2),
            make_candidate("b2", title="Baking", description="bread", category="Culinary", distance=0.9),
        ],
        **overrides,
    }


@pytest.mark.asyncio
async def test_evaluate_scores_and_sorts(fake_llm):
    result = await evaluate_node(make_state())
    by_id = {c["product_id"]: c["relevance_score"] for c in result["evaluated_candidates"]}
    assert set(by_id) == {"a1", "b2"}
    # a1: similar (0.8) + AI affinity (0.15) beats b2 (0.1)
    assert by_id["a1"] > by_id["b2"]
    assert [c["product_id"] for c in result["evaluated_candidates"]] == ["a1", "b2"]  # sorted desc


@pytest.mark.asyncio
async def test_evaluate_neutral_without_distance(fake_llm):
    result = await evaluate_node({
        "cognitive_model": {"inferred_intents": [], "category_affinity": [], "recent_categories": []},
        "retrieved_candidates": [make_candidate("x1", distance=None)],
    })
    assert result["evaluated_candidates"][0]["relevance_score"] == 0.5
    assert result["evaluated_candidates"][0]["evaluation_reasoning"]


@pytest.mark.asyncio
async def test_evaluate_boosts_category_affinity(fake_llm):
    result = await evaluate_node({
        "cognitive_model": {"inferred_intents": [], "category_affinity": ["Data"], "recent_categories": []},
        "retrieved_candidates": [
            make_candidate("data-1", category="Data", distance=0.5),
            make_candidate("other-1", category="Other", distance=0.5),
        ],
    })
    by_id = {c["product_id"]: c["relevance_score"] for c in result["evaluated_candidates"]}
    assert by_id["data-1"] > by_id["other-1"]


@pytest.mark.asyncio
async def test_evaluate_never_invents_ids_and_makes_no_llm_call(fake_llm):
    calls_before = fake_llm.chat_calls
    result = await evaluate_node(make_state())
    ids = {c["product_id"] for c in result["evaluated_candidates"]}
    assert ids <= {"a1", "b2"}
    assert fake_llm.chat_calls == calls_before  # deterministic — no LLM call