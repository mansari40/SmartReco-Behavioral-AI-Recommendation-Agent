"""
One-off smoke test for the reflect node. Tests both directions: does it
correctly accept a good narrative, and does it correctly catch a flawed
one (ignoring a stated objection)? A reflect node that approves
everything regardless of quality isn't actually doing its job.
"""
import asyncio

from app.agent.nodes.reflect import reflect_node


async def main():
    cognitive_model = {
        "session_arc": "User has been researching agentic AI systems and comparing course options.",
        "inferred_intents": ["building AI agents", "production RAG systems"],
        "stated_intents": ["agentic ai", "langgraph"],
        "decision_stage": "evaluation",
        "purchase_readiness": 0.65,
        "price_sensitivity": "medium",
        "detected_objections": ["time commitment"],
        "category_affinity": ["AI"],
    }

    print("Case 1: GOOD narrative (addresses the objection, fits the profile)")
    good_state = {
        "cognitive_model": cognitive_model,
        "narrative": (
            "You've clearly got momentum building toward agentic AI — and the good news is "
            "this course fits into a busy schedule with self-paced, modular lessons, so time "
            "commitment isn't the barrier it might seem."
        ),
        "persuasion_strategy": "scarcity_urgency",
        "regenerate_count": 0,
    }
    result_good = await reflect_node(good_state)
    print(f"  should_regenerate: {result_good['should_regenerate']}")
    print(f"  feedback: {result_good['reflection_feedback']!r}")
    print(f"  confidence: {result_good.get('confidence')}")

    print("\nCase 2: BAD narrative (completely ignores the time-commitment objection, generic)")
    bad_state = {
        "cognitive_model": cognitive_model,
        "narrative": (
            "Check out our courses! They're great for anyone interested in learning "
            "new skills. Sign up today and start your journey."
        ),
        "persuasion_strategy": "scarcity_urgency",
        "regenerate_count": 0,
    }
    result_bad = await reflect_node(bad_state)
    print(f"  should_regenerate: {result_bad['should_regenerate']}")
    print(f"  feedback: {result_bad['reflection_feedback']!r}")
    print(f"  regenerate_count: {result_bad['regenerate_count']}")


if __name__ == "__main__":
    asyncio.run(main())