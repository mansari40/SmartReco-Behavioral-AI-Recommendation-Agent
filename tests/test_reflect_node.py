"""reflect node — deterministic quality/grounding gate, no LLM. Always
accepts (never regenerates) so each agent run produces exactly one
generation; the output contract (should_regenerate / reflection_feedback /
regenerate_count / confidence) is unchanged."""
import pytest

from app.agent.nodes.reflect import reflect_node


def make_state(**overrides):
    return {
        "cognitive_model": {"detected_objections": ["time"], "decision_stage": "evaluation"},
        "narrative": "A narrative.",
        "persuasion_strategy": "objection_handling",
        "recommended_products": [{"product_id": "c1"}],
        "filtered_candidates": [{"product_id": "c1"}],
        "regenerate_count": 0,
        **overrides,
    }


@pytest.mark.asyncio
async def test_reflect_accepts_good_narrative(fake_llm):
    calls_before = fake_llm.chat_calls
    result = await reflect_node(make_state())
    assert result["should_regenerate"] is False
    assert result["regenerate_count"] == 0
    assert result["reflection_feedback"] == ""
    assert fake_llm.chat_calls == calls_before  # deterministic — no LLM call


@pytest.mark.asyncio
async def test_reflect_flags_empty_narrative(fake_llm):
    result = await reflect_node(make_state(narrative=""))
    assert result["should_regenerate"] is False
    assert "empty" in result["reflection_feedback"].lower()


@pytest.mark.asyncio
async def test_reflect_flags_ungrounded_product(fake_llm):
    result = await reflect_node(make_state(recommended_products=[{"product_id": "hallucinated"}]))
    assert result["should_regenerate"] is False
    assert "not among the evaluated" in result["reflection_feedback"]


@pytest.mark.asyncio
async def test_reflect_never_regenerates(fake_llm):
    """Regression: the old LLM-based reflect could request regeneration; the
    gate is now deterministic and a run always produces one narrative."""
    result = await reflect_node(make_state(regenerate_count=1))
    assert result["should_regenerate"] is False
    assert result["regenerate_count"] == 1  # unchanged