"""
One-off smoke test for the filter node — pure logic, no LLM/embedding
calls involved. Checks both branches: enough strong matches (threshold
path) and too few (fallback path).
"""
import asyncio

from app.agent.nodes.filter import filter_node


def make_candidate(title: str, score: float) -> dict:
    return {"product_id": title.lower().replace(" ", "-"), "title": title, "relevance_score": score}


async def main():
    print("Case 1: plenty of strong matches (should keep up to 4, threshold path)")
    state_a = {
        "evaluated_candidates": [
            make_candidate("Agentic AI Fundamentals", 0.9),
            make_candidate("Production RAG at Scale", 0.85),
            make_candidate("LangGraph for Multi-Agent Systems", 0.8),
            make_candidate("MLOps with Kubernetes", 0.75),
            make_candidate("Intro to Baking", 0.1),
        ],
    }
    result_a = await filter_node(state_a)
    print(f"  Kept: {[c['title'] for c in result_a['filtered_candidates']]}")
    print(f"  Alternatives: {result_a['alternatives_considered']}")

    print("\nCase 2: only one strong match (should fall back to MIN_RESULTS=2)")
    state_b = {
        "evaluated_candidates": [
            make_candidate("Agentic AI Fundamentals", 0.9),
            make_candidate("Intro to Baking", 0.2),
            make_candidate("Beginner Pottery", 0.15),
        ],
    }
    result_b = await filter_node(state_b)
    print(f"  Kept: {[c['title'] for c in result_b['filtered_candidates']]}")
    print(f"  Alternatives: {result_b['alternatives_considered']}")


if __name__ == "__main__":
    asyncio.run(main())