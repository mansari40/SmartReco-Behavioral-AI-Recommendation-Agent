"""
generate node: selects a persuasion strategy deterministically from the
cognitive model (not an LLM guess — this is what makes the "why this
strategy" claim in the transparency panel actually true), then asks the
LLM to write the persuasive narrative using that strategy and addressing
the user's detected objections directly. If a previous attempt was
rejected by the reflect node, incorporates that feedback so the retry
doesn't blindly repeat the same mistake.
"""
import json

from app.agent.state import AgentState
from app.services import llm_client


def _select_persuasion_strategy(cognitive_model: dict) -> tuple[str, str]:
    """Returns (strategy_name, instruction_for_llm). Rules are ordered —
    first match wins — so precedence is explicit and debuggable."""
    readiness = cognitive_model.get("purchase_readiness", 0.0)
    price_sensitivity = cognitive_model.get("price_sensitivity", "medium")
    decision_stage = cognitive_model.get("decision_stage", "awareness")
    objections = cognitive_model.get("detected_objections", [])

    if readiness > 0.6:
        return "scarcity_urgency", (
            "The user shows high purchase readiness. Use urgency/scarcity framing "
            "naturally (e.g. limited cohort spots, momentum they've already built) "
            "without being pushy or inventing fake deadlines."
        )
    if price_sensitivity == "high":
        return "social_proof", (
            "The user appears price-sensitive. Lead with social proof and "
            "concrete value (what they'll walk away able to do) rather than "
            "discounting language."
        )
    if decision_stage == "evaluation":
        return "authority_credibility", (
            "The user is actively comparing options. Emphasize depth/credibility "
            "of the content — what makes this specifically well-suited to what "
            "they're evaluating."
        )
    if objections:
        return "objection_handling", (
            f"Address this concern directly and early: {', '.join(objections)}. "
            "Reframe it rather than ignoring it."
        )
    return "curiosity_framing", (
        "The user is early in their journey. Spark curiosity about where this "
        "topic could take them rather than pushing for a decision."
    )


SYSTEM_PROMPT_TEMPLATE = """You are writing a personalized product recommendation message.

USER PROFILE:
{profile}

PERSUASION STRATEGY TO USE: {strategy_name}
INSTRUCTION: {strategy_instruction}

PRODUCTS TO RECOMMEND:
{products}

RULES:
1. Write a short narrative (under 100 words) explaining why these products fit this user's journey.
2. Use the assigned persuasion strategy naturally — don't name it explicitly, just apply it.
3. If the user has detected objections, address at least one of them somewhere in the message.
4. Do not invent facts about the products beyond what's given.
5. Then, for each product, write one specific 1-sentence reason it fits THIS user.

Return ONLY a JSON object with this exact shape, no other text:
{{
  "narrative": "the main message",
  "product_reasons": [{{"product_id": "...", "reason": "..."}}],
  "reasoning_chain": ["step 1 of your reasoning", "step 2", "step 3"]
}}"""


def _safe_parse_generation(raw: str, filtered_candidates: list[dict]) -> dict:
    fallback_products = [{"product_id": c["product_id"], "reason": c["title"]} for c in filtered_candidates]
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        narrative = data.get("narrative")
        product_reasons = data.get("product_reasons")
        reasoning_chain = data.get("reasoning_chain")

        if not isinstance(narrative, str) or not narrative.strip():
            narrative = "Based on your recent activity, here are some courses that fit your interests."
        if not isinstance(product_reasons, list) or not product_reasons:
            product_reasons = fallback_products
        if not isinstance(reasoning_chain, list):
            reasoning_chain = []

        return {"narrative": narrative, "product_reasons": product_reasons, "reasoning_chain": reasoning_chain}
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {
            "narrative": "Based on your recent activity, here are some courses that fit your interests.",
            "product_reasons": fallback_products,
            "reasoning_chain": ["fallback: generation response could not be parsed"],
        }


async def generate_node(state: AgentState) -> dict:
    cognitive_model = state["cognitive_model"]
    filtered_candidates = state.get("filtered_candidates", [])

    if not filtered_candidates:
        return {
            "narrative": "We're still learning what you're interested in — keep exploring!",
            "recommended_products": [],
            "persuasion_strategy": "none",
            "confidence": 0.0,
            "reasoning_chain": ["no candidates passed the relevance threshold"],
        }

    strategy_name, strategy_instruction = _select_persuasion_strategy(cognitive_model)

    products_text = "\n".join(
        f"- product_id: {c['product_id']}, title: {c['title']}, description: {c['description']}, price: {c['price']}"
        for c in filtered_candidates
    )

    feedback_note = ""
    if state.get("reflection_feedback"):
        feedback_note = (
            f"\n\nIMPORTANT: A previous attempt at this message was reviewed and rejected. "
            f"Fix this specific issue: {state['reflection_feedback']}"
        )

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        profile=json.dumps(cognitive_model, indent=2),
        strategy_name=strategy_name,
        strategy_instruction=strategy_instruction,
        products=products_text,
    ) + feedback_note

    raw_reply = await llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        response_format_json=True,
        temperature=0.7,
    )

    parsed = _safe_parse_generation(raw_reply, filtered_candidates)

    avg_relevance = sum(c["relevance_score"] for c in filtered_candidates) / len(filtered_candidates)

    return {
        "narrative": parsed["narrative"],
        "recommended_products": parsed["product_reasons"],
        "persuasion_strategy": strategy_name,
        "confidence": round(avg_relevance, 2),
        "reasoning_chain": parsed["reasoning_chain"],
    }