"""
reflect node: critiques the just-generated narrative against the user's
actual profile, using specific pointed questions rather than a vague
"is this good?" (which models tend to reflexively approve). Increments
regenerate_count on a "no" — the graph's conditional edge (in graph.py)
enforces the MAX_REGENERATIONS cap, this node just reports its honest
verdict.
"""
import json

from app.agent.state import AgentState
from app.services import llm_client

SYSTEM_PROMPT = """You are critiquing a generated product recommendation message before
it gets shown to a real user. Be honest and specific — don't rubber-stamp it.

USER PROFILE:
{profile}

GENERATED NARRATIVE:
{narrative}

PERSUASION STRATEGY USED: {strategy}

Answer these questions honestly:
1. Does the narrative actually address the user's detected objections (if any), or just ignore them?
2. Is the persuasion strategy applied appropriately for this user's decision stage, or does it feel mismatched/pushy?
3. Would a real person with this profile plausibly find this message convincing, or does it feel generic/could apply to any user?
4. Does it invent any claims not supported by the product data?

Return ONLY a JSON object with this exact shape, no other text:
{{
  "should_regenerate": true or false,
  "feedback": "specific, actionable feedback for what to fix, or empty string if no issues",
  "confidence": float 0.0-1.0 (your confidence this message will actually work on this user)
}}

Only set should_regenerate to true if there's a real, specific problem — not for minor stylistic preference."""


def _safe_parse_reflection(raw: str) -> dict:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return {
            "should_regenerate": bool(data.get("should_regenerate", False)),
            "feedback": data.get("feedback", "") if isinstance(data.get("feedback"), str) else "",
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.7)))),
        }
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        # If reflection itself fails to parse, default to accepting the
        # narrative rather than looping — a parse failure here shouldn't
        # block the user from getting a recommendation at all.
        return {"should_regenerate": False, "feedback": "", "confidence": 0.7}


async def reflect_node(state: AgentState) -> dict:
    current_count = state.get("regenerate_count", 0)

    prompt = SYSTEM_PROMPT.format(
        profile=json.dumps(state["cognitive_model"], indent=2),
        narrative=state.get("narrative", ""),
        strategy=state.get("persuasion_strategy", ""),
    )

    raw_reply = await llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        response_format_json=True,
        temperature=0.3,
    )

    verdict = _safe_parse_reflection(raw_reply)

    return {
        "should_regenerate": verdict["should_regenerate"],
        "reflection_feedback": verdict["feedback"],
        "regenerate_count": current_count + (1 if verdict["should_regenerate"] else 0),
        "confidence": verdict["confidence"] if not verdict["should_regenerate"] else state.get("confidence", 0.0),
    }