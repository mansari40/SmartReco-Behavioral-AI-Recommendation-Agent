"""filter node — pure logic, no LLM. Threshold path keeps top matches;
fallback path guarantees a minimum number of recommendations."""
import pytest

from app.agent.nodes.filter import filter_node, MAX_RESULTS, MIN_RESULTS


def make_candidate(title: str, score: float) -> dict:
    return {"product_id": title.lower().replace(" ", "-"), "title": title, "relevance_score": score}


@pytest.mark.asyncio
async def test_threshold_path_keeps_top_matches():
    state = {"evaluated_candidates": [
        make_candidate("Agentic AI Fundamentals", 0.9),
        make_candidate("Production RAG at Scale", 0.85),
        make_candidate("LangGraph Systems", 0.8),
        make_candidate("MLOps with Kubernetes", 0.75),
        make_candidate("Intro to Baking", 0.1),
    ]}
    result = await filter_node(state)
    assert [c["title"] for c in result["filtered_candidates"]] == [
        "Agentic AI Fundamentals", "Production RAG at Scale", "LangGraph Systems", "MLOps with Kubernetes",
    ]
    assert len(result["filtered_candidates"]) <= MAX_RESULTS
    assert any("not included" in a for a in result["alternatives_considered"])


@pytest.mark.asyncio
async def test_fallback_path_guarantees_minimum():
    state = {"evaluated_candidates": [
        make_candidate("Agentic AI Fundamentals", 0.9),
        make_candidate("Intro to Baking", 0.2),
        make_candidate("Beginner Pottery", 0.15),
    ]}
    result = await filter_node(state)
    assert len(result["filtered_candidates"]) == MIN_RESULTS


@pytest.mark.asyncio
async def test_empty_input():
    result = await filter_node({"evaluated_candidates": []})
    assert result["filtered_candidates"] == []