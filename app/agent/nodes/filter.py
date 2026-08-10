"""
filter node: keeps the top candidates from evaluate's sorted list, using a
score threshold rather than a fixed count — a strongly-matching user might
get 4 solid recommendations, a weakly-matching one might only get 2. No
LLM call needed here; evaluate already did the scoring work.
"""
from app.agent.state import AgentState

MIN_RELEVANCE_SCORE = 0.5
MIN_RESULTS = 2
MAX_RESULTS = 4


async def filter_node(state: AgentState) -> dict:
    evaluated = state.get("evaluated_candidates", [])

    above_threshold = [c for c in evaluated if c["relevance_score"] >= MIN_RELEVANCE_SCORE]

    if len(above_threshold) >= MIN_RESULTS:
        kept = above_threshold[:MAX_RESULTS]
    else:
        # Not enough strong matches — fall back to the best available
        # candidates anyway (evaluated is already sorted), so the user
        # still gets *something* rather than an empty recommendation.
        kept = evaluated[:MIN_RESULTS]

    rejected = [c for c in evaluated if c not in kept]

    return {
        "filtered_candidates": kept,
        "alternatives_considered": [
            f"{c['title']} (score: {c['relevance_score']:.2f}) — not included"
            for c in rejected
        ],
    }