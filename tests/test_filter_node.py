"""filter node — pure logic, no LLM. Keeps candidates that clear the
relevance floor; MIN_RESULTS is a target, never enforced by force-filling
with irrelevant candidates."""
import pytest

from app.agent.nodes.filter import filter_node, MAX_RESULTS, RELEVANCE_FLOOR


def make_candidate(title: str, score: float) -> dict:
    return {"product_id": title.lower().replace(" ", "-"), "title": title, "relevance_score": score}


@pytest.mark.asyncio
async def test_floor_path_keeps_top_matches():
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
async def test_single_candidate_above_floor_is_kept_alone():
    """MIN_RESULTS is a target, not a requirement — one valid candidate is
    better than two, one of which is irrelevant."""
    state = {"evaluated_candidates": [
        make_candidate("Agentic AI Fundamentals", 0.9),
        make_candidate("Intro to Baking", 0.1),
    ]}
    result = await filter_node(state)
    assert [c["title"] for c in result["filtered_candidates"]] == ["Agentic AI Fundamentals"]


@pytest.mark.asyncio
async def test_near_zero_candidates_are_not_forced_in():
    """Score-0 / near-zero candidates must never be force-filled to reach
    MIN_RESULTS — an empty result is correct when nothing clears the floor."""
    state = {"evaluated_candidates": [
        make_candidate("Data Quality Engineering", 0.02),
        make_candidate("Intro to Baking", 0.0),
    ]}
    result = await filter_node(state)
    assert result["filtered_candidates"] == []


@pytest.mark.asyncio
async def test_below_floor_candidates_are_rejected_even_when_few():
    state = {"evaluated_candidates": [
        make_candidate("Agentic AI Fundamentals", 0.15),
        make_candidate("Production RAG at Scale", 0.05),
    ]}
    result = await filter_node(state)
    assert result["filtered_candidates"] == []


@pytest.mark.asyncio
async def test_floor_boundary_is_inclusive():
    state = {"evaluated_candidates": [make_candidate("Agentic AI Fundamentals", RELEVANCE_FLOOR)]}
    result = await filter_node(state)
    assert [c["title"] for c in result["filtered_candidates"]] == ["Agentic AI Fundamentals"]


@pytest.mark.asyncio
async def test_empty_input():
    result = await filter_node({"evaluated_candidates": []})
    assert result["filtered_candidates"] == []