"""
One-off smoke test for the evaluate node. Feeds it two very different
candidates (one clearly matching the user's interest, one clearly not)
alongside a cognitive model, and checks that the LLM actually
discriminates between them rather than scoring everything the same.
"""
import asyncio

from app.agent.nodes.evaluate import evaluate_node


async def main():
    fake_state = {
        "user_id": "test-user",
        "cognitive_model": {
            "session_arc": "User has been searching for and viewing agentic AI and LangGraph content repeatedly.",
            "inferred_intents": ["agentic AI", "building AI agents", "RAG pipelines"],
            "stated_intents": ["agentic ai", "langgraph"],
            "decision_stage": "evaluation",
            "purchase_readiness": 0.6,
            "price_sensitivity": "medium",
            "detected_objections": ["time commitment"],
            "category_affinity": ["AI"],
        },
        "retrieved_candidates": [
            {
                "product_id": "ai-course-1",
                "title": "Agentic AI Fundamentals",
                "description": "Learn to build reasoning agents with LangGraph and RAG pipelines.",
                "category": "AI",
                "price": 49.99,
                "distance": 0.1,
            },
            {
                "product_id": "baking-course-1",
                "title": "Introduction to Baking",
                "description": "Learn to bake bread and pastries at home.",
                "category": "Culinary",
                "price": 19.99,
                "distance": 0.9,
            },
        ],
    }

    print("Running evaluate_node...")
    result = await evaluate_node(fake_state)

    print("\nEvaluated candidates (sorted by relevance_score):")
    for c in result["evaluated_candidates"]:
        print(f"  - {c['title']}: score={c['relevance_score']:.2f} | {c['evaluation_reasoning']}")


if __name__ == "__main__":
    asyncio.run(main())