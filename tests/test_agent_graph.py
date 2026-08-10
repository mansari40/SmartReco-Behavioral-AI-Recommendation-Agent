"""
One-off smoke test for the agent graph's shape — proves nodes are wired
correctly and the reflect->generate loop / store->END path work, before
any real logic exists in the nodes themselves.
"""
import asyncio

from app.agent.graph import agent_graph


async def main():
    result = await agent_graph.ainvoke({"user_id": "test-user-123", "trigger_reason": "manual test"})
    print("\nFinal state keys:", list(result.keys()))
    print("Narrative:", result.get("narrative"))


if __name__ == "__main__":
    asyncio.run(main())