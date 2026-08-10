"""
One-off smoke test for the generate node — the most subjective one to
evaluate, since it's real written copy, not just structured data. Run
this a couple of times; temperature=0.7 means output will vary between
runs, which is expected and fine.
"""
import asyncio

from app.agent.nodes.generate import generate_node


async def main():
    fake_state = {
        "user_id": "test-user",
        "cognitive_model": {
            "session_arc": "User has been researching agentic AI systems, viewing course details repeatedly, and comparing options.",
            "inferred_intents": ["building AI agents", "production RAG systems"],
            "stated_intents": ["agentic ai", "langgraph"],
            "decision_stage": "evaluation",
            "purchase_readiness": 0.65,
            "price_sensitivity": "medium",
            "detected_objections": ["time commitment"],
            "category_affinity": ["AI"],
        },
        "filtered_candidates": [
            {
                "product_id": "ai-course-1",
                "title": "Agentic AI Fundamentals",
                "description": "Learn to build reasoning agents with LangGraph and RAG pipelines. Self-paced, 6 modules.",
                "category": "AI",
                "price": 49.99,
                "relevance_score": 0.9,
            },
            {
                "product_id": "rag-course-1",
                "title": "Production RAG at Scale",
                "description": "Advanced retrieval-augmented generation patterns for real-world systems.",
                "category": "AI",
                "price": 79.99,
                "relevance_score": 0.75,
            },
        ],
    }

    print("Running generate_node...\n")
    result = await generate_node(fake_state)

    print(f"Persuasion strategy selected: {result['persuasion_strategy']}")
    print(f"Confidence: {result['confidence']}\n")
    print("Narrative:")
    print(f"  {result['narrative']}\n")
    print("Recommended products:")
    for p in result["recommended_products"]:
        print(f"  - {p}")
    print("\nReasoning chain:")
    for step in result["reasoning_chain"]:
        print(f"  - {step}")


if __name__ == "__main__":
    asyncio.run(main())