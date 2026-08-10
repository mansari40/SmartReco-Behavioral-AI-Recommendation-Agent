"""
evaluate node: scores each retrieved candidate against the user's
cognitive model, answering "would this actually convince THIS user" —
not just "is this semantically similar" (that's already been handled by
retrieve). One LLM call scores all candidates together, not one call per
candidate, to stay within the efficiency requirement.
"""
import json

from app.agent.state import AgentState
from app.services import llm_client

SYSTEM_PROMPT = """You are evaluating product candidates for a specific user, based on
their behavioral profile. For each candidate, judge how well it actually
fits this user's demonstrated interest and readiness — not just topical
similarity.

Return ONLY a JSON object with this exact shape, no other text:
{
  "scores": [
    {"product_id": "...", "relevance_score": float 0.0-1.0, "reasoning": "one sentence"}
  ]
}

Score based on: does this match their inferred intents (not just stated
ones)? Does it fit their decision stage? Would it plausibly address any
detected objections? A topically-similar product that ignores their
decision stage or objections should score lower than one that fits both."""


def _safe_parse_scores(raw: str, candidate_ids: set[str]) -> dict[str, dict]:
    """Returns {product_id: {"relevance_score": float, "reasoning": str}}.
    Falls back to a neutral 0.5 score for any candidate the LLM didn't
    return a valid entry for, rather than dropping it silently."""
    scores: dict[str, dict] = {}
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        for entry in data.get("scores", []):
            pid = entry.get("product_id")
            score = entry.get("relevance_score")
            if pid in candidate_ids and isinstance(score, (int, float)):
                scores[pid] = {
                    "relevance_score": max(0.0, min(1.0, float(score))),
                    "reasoning": entry.get("reasoning", ""),
                }
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass  # fall through to defaults below

    for pid in candidate_ids:
        scores.setdefault(pid, {"relevance_score": 0.5, "reasoning": "no evaluation returned"})

    return scores


async def evaluate_node(state: AgentState) -> dict:
    candidates = state.get("retrieved_candidates", [])
    if not candidates:
        return {"evaluated_candidates": []}

    cognitive_model = state["cognitive_model"]
    candidate_ids = {c["product_id"] for c in candidates}

    candidates_text = "\n".join(
        f"- product_id: {c['product_id']}, title: {c['title']}, "
        f"category: {c['category']}, description: {c['description']}"
        for c in candidates
    )

    user_prompt = (
        f"USER PROFILE:\n{json.dumps(cognitive_model, indent=2)}\n\n"
        f"CANDIDATES:\n{candidates_text}"
    )

    raw_reply = await llm_client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format_json=True,
        temperature=0.3,
    )

    scores = _safe_parse_scores(raw_reply, candidate_ids)

    evaluated = []
    for c in candidates:
        score_info = scores[c["product_id"]]
        evaluated.append({
            **c,
            "relevance_score": score_info["relevance_score"],
            "evaluation_reasoning": score_info["reasoning"],
        })

    evaluated.sort(key=lambda c: c["relevance_score"], reverse=True)
    return {"evaluated_candidates": evaluated}