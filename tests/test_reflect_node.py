"""reflect node — honest critique that can request regeneration, bounded by
the graph's retry cap; parse failures must never block a recommendation."""
import pytest

from app.agent.nodes.reflect import reflect_node


def make_state(**overrides):
    return {
        "cognitive_model": {"detected_objections": ["time"], "decision_stage": "evaluation"},
        "narrative": "A narrative.",
        "persuasion_strategy": "objection_handling",
        "regenerate_count": 0,
        **overrides,
    }


@pytest.mark.asyncio
async def test_reflect_accepts_good_narrative(fake_llm):
    result = await reflect_node(make_state())
    assert result["should_regenerate"] is False
    assert result["regenerate_count"] == 0


@pytest.mark.asyncio
async def test_reflect_can_request_regeneration(fake_llm):
    fake_llm.responses = {"critiquing a generated": json.dumps(
        {"should_regenerate": True, "feedback": "ignores the time objection", "confidence": 0.3}
    )}
    result = await reflect_node(make_state())
    assert result["should_regenerate"] is True
    assert result["regenerate_count"] == 1
    assert "time" in result["reflection_feedback"]


@pytest.mark.asyncio
async def test_reflect_parse_failure_accepts(fake_llm):
    fake_llm.responses = {"critiquing a generated": "not json"}
    result = await reflect_node(make_state())
    assert result["should_regenerate"] is False  # never block on a parse failure
    assert result["reflection_feedback"] == ""


import json  # noqa: E402