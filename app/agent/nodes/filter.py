"""
filter node: keeps the candidates from evaluate's sorted list that clear a
relevance floor — a strongly-matching user might get 4 solid
recommendations, a weakly-matching one might only get 1, and one with no
genuine matches gets NONE. MIN_RESULTS is a target, NOT a requirement:
never invent a recommendation by filling slots with irrelevant candidates.
No LLM call needed here; evaluate already did the scoring work.
"""
from app.agent.state import AgentState

RELEVANCE_FLOOR = 0.20
MIN_RESULTS = 2  # target only — never enforced by force-filling below-floor candidates
MAX_RESULTS = 4


async def filter_node(state: AgentState) -> dict:
    evaluated = state.get("evaluated_candidates", [])

    above_floor = [c for c in evaluated if c["relevance_score"] >= RELEVANCE_FLOOR]

    kept = above_floor[:MAX_RESULTS]

    rejected = [c for c in evaluated if c not in kept]

    return {
        "filtered_candidates": kept,
        "alternatives_considered": [
            f"{c['title']} (score: {c['relevance_score']:.2f}) — not included"
            for c in rejected
        ],
    }