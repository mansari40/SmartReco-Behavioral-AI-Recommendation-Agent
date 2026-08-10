"""
Runs the recommendation agent graph for a given user. Wrapped separately
from graph.py so it can be invoked as a FastAPI background task with
proper error handling — a background task that raises an unhandled
exception fails silently from the client's perspective (the HTTP response
already went out), so this must log clearly rather than let failures
vanish unnoticed.
"""
import logging

from app.agent.graph import agent_graph

logger = logging.getLogger(__name__)


async def run_agent_for_user(user_id: str, trigger_reason: str) -> None:
    try:
        logger.info("Agent run starting for user=%s reason=%s", user_id, trigger_reason)
        await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": trigger_reason})
        logger.info("Agent run completed for user=%s", user_id)
    except Exception:
        logger.exception("Agent run failed for user=%s", user_id)