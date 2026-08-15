"""
evaluate node: scores each retrieved candidate against the user's cognitive
model, answering "would this actually convince THIS user" — not just "is
this semantically similar" (that's already been handled by retrieve).

Deterministic — no LLM call — so the agent makes exactly one generation call
per run (the generate node). Scoring blends the semantic similarity the
vector store already computed (from the distance) with deterministic signals
from the cognitive model: category affinity and inferred-intent keyword
overlap. This keeps the full agentic pipeline but moves scoring out of the
LLM, cutting a redundant call and its latency.
"""
from app.agent.state import AgentState


def _similarity(candidate: dict) -> float:
    """Semantic similarity derived from the vector-store distance
    (0 = identical, 1 = orthogonal). Candidates without a distance get a
    neutral 0.5 so nothing is silently dropped."""
    distance = candidate.get("distance")
    if distance is None:
        return 0.5
    try:
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except (TypeError, ValueError):
        return 0.5


def _category_bonus(candidate: dict, cognitive_model: dict) -> float:
    """+0.15 when the candidate's category is one the user has actually
    engaged with (recent views or accumulated category affinity)."""
    categories = set(cognitive_model.get("recent_categories", [])) | set(
        cognitive_model.get("category_affinity", [])
    )
    return 0.15 if candidate.get("category") in categories else 0.0


def _intent_bonus(candidate: dict, cognitive_model: dict) -> float:
    """+0.1 per inferred intent whose keywords appear in the candidate's
    title/description, capped at +0.3."""
    text = f"{candidate.get('title', '')} {candidate.get('description', '')}".lower()
    overlap = 0
    for intent in cognitive_model.get("inferred_intents", []):
        if isinstance(intent, str) and intent.strip() and intent.lower() in text:
            overlap += 1
    return min(overlap * 0.1, 0.3)


async def evaluate_node(state: AgentState) -> dict:
    candidates = state.get("retrieved_candidates", [])
    if not candidates:
        return {"evaluated_candidates": []}

    cognitive_model = state["cognitive_model"]

    evaluated = []
    for c in candidates:
        similarity = _similarity(c)
        category_bonus = _category_bonus(c, cognitive_model)
        intent_bonus = _intent_bonus(c, cognitive_model)
        score = max(0.0, min(1.0, similarity + category_bonus + intent_bonus))

        sources = []
        if category_bonus:
            sources.append(c.get("category", "your recent interests"))
        if intent_bonus:
            sources.append("your inferred interests")
        reason = "topically similar to your activity" if not sources else "matches " + ", ".join(sources)

        evaluated.append({
            **c,
            "relevance_score": round(score, 3),
            "evaluation_reasoning": reason,
        })

    evaluated.sort(key=lambda c: c["relevance_score"], reverse=True)
    return {"evaluated_candidates": evaluated}