"""
reflect node: a deterministic quality gate on the generated recommendation —
no LLM call. Verifies the output is complete (non-empty narrative) and
grounded (every recommended product is a real filtered candidate), then
always accepts. It never requests regeneration, so the graph produces
exactly one narrative per run: the agent's only LLM *generation* call is the
generate node. The regeneration loop mechanism in graph.py is retained for
future use but is intentionally never activated, keeping LLM cost bounded.
"""
from app.agent.state import AgentState


async def reflect_node(state: AgentState) -> dict:
    narrative = state.get("narrative", "")
    recommended = state.get("recommended_products", [])
    grounded_ids = {c["product_id"] for c in state.get("filtered_candidates", [])}

    feedback = ""
    if not narrative or not str(narrative).strip():
        feedback = "generated narrative was empty"
    elif any(p.get("product_id") not in grounded_ids for p in recommended):
        feedback = "recommended product was not among the evaluated candidates"

    return {
        "should_regenerate": False,
        "reflection_feedback": feedback,
        "regenerate_count": state.get("regenerate_count", 0),
        "confidence": state.get("confidence", 0.0),
    }