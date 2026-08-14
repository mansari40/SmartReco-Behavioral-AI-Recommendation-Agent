"""evaluate node — scores candidates against the cognitive model; malformed
or sparse LLM output must degrade to neutral scores, never crash."""
import json

import pytest

from app.agent.nodes.evaluate import evaluate_node


@pytest.mark.asyncio
async def test_evaluate_scores_and_sorts(fake_llm):
    state = {
        "cognitive_model": {"inferred_intents": ["building AI agents"], "decision_stage": "evaluation"},
        "retrieved_candidates": [
            {"product_id": "a1", "title": "AI Course", "category": "AI", "description": "agents", "price": 50},
            {"product_id": "b2", "title": "Baking", "category": "Culinary", "description": "bread", "price": 20},
        ],
    }
    result = await evaluate_node(state)
    assert len(result["evaluated_candidates"]) == 2
    # mocked LLM returns 0.9 for everything → both kept, sorted
    assert all(c["relevance_score"] == 0.9 for c in result["evaluated_candidates"])


@pytest.mark.asyncio
async def test_evaluate_falls_back_on_garbage_llm(fake_llm):
    fake_llm.responses = {"evaluating product candidates": "this is not json at all"}
    state = {
        "cognitive_model": {"inferred_intents": ["x"]},
        "retrieved_candidates": [
            {"product_id": "a1", "title": "A", "category": "AI", "description": "d", "price": 1},
        ],
    }
    result = await evaluate_node(state)
    assert result["evaluated_candidates"][0]["relevance_score"] == 0.5  # neutral fallback
    assert "no evaluation returned" in result["evaluated_candidates"][0]["evaluation_reasoning"]


@pytest.mark.asyncio
async def test_evaluate_ignores_invented_product_ids(fake_llm):
    fake_llm.responses = {"evaluating product candidates": json.dumps({
        "scores": [{"product_id": "hallucinated-99", "relevance_score": 1.0, "reasoning": "fake"}]
    })}
    state = {
        "cognitive_model": {"inferred_intents": ["x"]},
        "retrieved_candidates": [
            {"product_id": "real-1", "title": "Real", "category": "AI", "description": "d", "price": 1},
        ],
    }
    result = await evaluate_node(state)
    assert result["evaluated_candidates"][0]["product_id"] == "real-1"
    assert result["evaluated_candidates"][0]["relevance_score"] == 0.5  # invented id dropped