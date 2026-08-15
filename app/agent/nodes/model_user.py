"""
model_user node: reads a user's recent events, asks the LLM to infer their
interests/intent/decision-stage, and persists the result as a
UserCognitiveModel row. This is the "theory of mind" step — everything
downstream (retrieve, generate) reasons off this structured summary rather
than raw events directly.
"""
import json
from collections import Counter

from sqlalchemy import select

from app.agent.state import AgentState
from app.db.session import AsyncSessionLocal
from app.models.cognitive_model import DecisionStage, PriceSensitivity
from app.models.event import Event, EventType
from app.models.product import Product
from app.services import llm_client
from app.services.trigger_service import get_or_create_cognitive_model

MAX_EVENTS_CONSIDERED = 50

SYSTEM_PROMPT = """You are a user-behavior analyst for an online course marketplace.
Given a user's recent activity and their current profile, update their profile.

Return ONLY a JSON object with exactly these fields, no other text:
{
  "stated_intents": [list of strings — things the user explicitly searched for],
  "inferred_intents": [list of strings — deeper interests you infer beyond the literal searches],
  "decision_stage": one of "awareness", "interest", "evaluation", "decision", "post_purchase",
  "purchase_readiness": float between 0.0 and 1.0,
  "price_sensitivity": one of "low", "medium", "high",
  "detected_objections": [list of strings — concerns the behavior suggests, e.g. "price", "time commitment"],
  "brand_affinity": [list of strings],
  "category_affinity": [list of strings — product categories they've shown interest in],
  "session_arc": "one sentence describing the narrative of their browsing session so far"
}

Base decision_stage and purchase_readiness on signal strength: a single
search is weak evidence (awareness/low readiness); repeated views of the
same category or product, or longer time_spent, are stronger evidence
(interest/evaluation, higher readiness)."""


async def _format_events_for_prompt(db, events: list[Event]) -> str:
    product_ids = {e.product_id for e in events if e.product_id}
    products_by_id: dict[str, Product] = {}
    if product_ids:
        result = await db.execute(select(Product).where(Product.id.in_(product_ids)))
        products_by_id = {p.id: p for p in result.scalars().all()}

    categories = []
    search_queries = []
    repeated_views = Counter()
    total_time_spent = 0

    lines = []
    for e in events:
        detail = f"- {e.event_type.value}"
        if e.product_id and e.product_id in products_by_id:
            p = products_by_id[e.product_id]
            detail += f" on '{p.title}' (category: {p.category})"
            if e.event_type == EventType.PRODUCT_VIEW:
                categories.append(p.category)
                repeated_views[p.title] += 1
        if e.event_type == EventType.SEARCH and e.event_metadata.get("query"):
            query = e.event_metadata["query"].strip()
            search_queries.append(query)
        if e.event_type == EventType.TIME_SPENT:
            total_time_spent += int(e.event_metadata.get("seconds", 0) or 0)
        if e.event_metadata:
            detail += f" | metadata: {json.dumps(e.event_metadata)}"
        lines.append(detail)

    if search_queries:
        lines.append(f"- recent search queries: {', '.join(dict.fromkeys(search_queries))}")
    if categories:
        top_categories = [c for c, _ in Counter(categories).most_common(3)]
        lines.append(f"- category affinity signals: {', '.join(top_categories)}")
    repeated = [title for title, count in repeated_views.items() if count > 1]
    if repeated:
        lines.append(f"- repeated interest in: {', '.join(repeated)}")
    if total_time_spent:
        lines.append(f"- total page engagement: {round(total_time_spent / 60, 1)} minutes")

    return "\n".join(lines) if lines else "(no events)"


def _safe_parse_cognitive_update(raw: str, fallback: dict) -> dict:
    """Defensive parse — the LLM should return clean JSON (we ask for it
    explicitly), but never trust that unconditionally. Falls back to the
    previous values for any field that's missing or malformed."""
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        return fallback

    result = dict(fallback)
    for key in (
        "stated_intents", "inferred_intents", "detected_objections",
        "brand_affinity", "category_affinity",
    ):
        if isinstance(data.get(key), list):
            result[key] = data[key]

    if data.get("decision_stage") in [s.value for s in DecisionStage]:
        result["decision_stage"] = data["decision_stage"]

    if data.get("price_sensitivity") in [s.value for s in PriceSensitivity]:
        result["price_sensitivity"] = data["price_sensitivity"]

    if isinstance(data.get("purchase_readiness"), (int, float)):
        result["purchase_readiness"] = max(0.0, min(1.0, float(data["purchase_readiness"])))

    if isinstance(data.get("session_arc"), str):
        result["session_arc"] = data["session_arc"]

    return result


async def _extract_recent_signals(events: list[Event], products_by_id: dict) -> tuple[list[str], list[str]]:
    """Deterministic recency extraction — no LLM. Returns (recent_searches,
    recent_categories) in chronological order (oldest first), so retrieval
    can favor the user's most current intent over accumulated history."""
    recent_searches = [
        e.event_metadata.get("query", "").strip()
        for e in events
        if e.event_type == EventType.SEARCH and e.event_metadata.get("query", "").strip()
    ]
    recent_categories = []
    for e in events:
        if e.event_type == EventType.PRODUCT_VIEW and e.product_id:
            p = products_by_id.get(e.product_id)
            if p and p.category not in recent_categories:
                recent_categories.append(p.category)
    return recent_searches[-5:], recent_categories[-3:]


async def model_user_node(state: AgentState) -> dict:
    user_id = state["user_id"]

    async with AsyncSessionLocal() as db:
        model = await get_or_create_cognitive_model(db, user_id)

        events_result = await db.execute(
            select(Event)
            .where(Event.user_id == user_id)
            .order_by(Event.created_at.desc())
            .limit(MAX_EVENTS_CONSIDERED)
        )
        events = list(events_result.scalars().all())
        # Sort chronologically (oldest first). A stable sort keeps insertion
        # order intact when events share the same created_at timestamp.
        events.sort(key=lambda e: e.created_at)
        total_event_count = len(events)  # approximation; fine for the trigger's purposes

        events_text = await _format_events_for_prompt(db, events)

        product_ids = {e.product_id for e in events if e.product_id}
        products_by_id: dict[str, Product] = {}
        if product_ids:
            result = await db.execute(select(Product).where(Product.id.in_(product_ids)))
            products_by_id = {p.id: p for p in result.scalars().all()}

        recent_searches, recent_categories = await _extract_recent_signals(events, products_by_id)

        current_profile = {
            "stated_intents": model.stated_intents,
            "inferred_intents": model.inferred_intents,
            "decision_stage": model.decision_stage.value,
            "purchase_readiness": model.purchase_readiness,
            "price_sensitivity": model.price_sensitivity.value,
            "detected_objections": model.detected_objections,
            "brand_affinity": model.brand_affinity,
            "category_affinity": model.category_affinity,
            "session_arc": model.session_arc,
        }

        user_prompt = (
            f"CURRENT PROFILE:\n{json.dumps(current_profile, indent=2)}\n\n"
            f"RECENT ACTIVITY:\n{events_text}"
        )

        raw_reply = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format_json=True,
            temperature=0.3,
        )

        updated = _safe_parse_cognitive_update(raw_reply, fallback=current_profile)

        model.stated_intents = updated["stated_intents"]
        model.inferred_intents = updated["inferred_intents"]
        model.decision_stage = DecisionStage(updated["decision_stage"])
        model.purchase_readiness = updated["purchase_readiness"]
        model.price_sensitivity = PriceSensitivity(updated["price_sensitivity"])
        model.detected_objections = updated["detected_objections"]
        model.brand_affinity = updated["brand_affinity"]
        model.category_affinity = updated["category_affinity"]
        model.recent_searches = recent_searches
        model.recent_categories = recent_categories
        model.session_arc = updated["session_arc"]
        model.last_event_count_at_update = total_event_count

        await db.commit()

        updated["recent_searches"] = recent_searches
        updated["recent_categories"] = recent_categories

        return {"cognitive_model": updated}