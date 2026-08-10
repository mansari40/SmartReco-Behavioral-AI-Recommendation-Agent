"""
One-off smoke test for the Mesh API client. Not part of the app —
delete this file once you've confirmed it works.
"""
import asyncio

from app.services.llm_client import chat_completion, get_embedding

async def main():
    print("Testing chat_completion...")
    reply = await chat_completion(
        messages=[{"role": "user", "content": "Reply with exactly one word: pong"}]
    )
    print("Chat reply:", repr(reply))

    print("\nTesting get_embedding...")
    vector = await get_embedding("agentic AI course")
    print("Embedding length:", len(vector))
    print("First 5 values:", vector[:5])


if __name__ == "__main__":
    asyncio.run(main())