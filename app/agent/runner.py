"""
Runs the recommendation agent graph for a given user. Wrapped separately
from graph.py so it can be invoked as a FastAPI background task with
proper error handling — a background task that raises an unhandled
exception fails silently from the client's perspective (the HTTP response
already went out), so this must log clearly rather than let failures
vanish unnoticed.

Also enforces a per-user in-flight guard: only one agent run per user may be
executing at a time. Rapid event batches (or the event trigger racing the
daily digest) can otherwise launch concurrent runs for the same user — each
making its own LLM calls and writing a duplicate recommendation.
"""
import asyncio
import logging

from app.agent.graph import agent_graph

logger = logging.getLogger(__name__)

_in_flight: set[str] = set()
_in_flight_lock = asyncio.Lock()


async def _claim(user_id: str) -> bool:
    """True if this caller won the slot for the user; False if a run for the
    same user is already in flight."""
    async with _in_flight_lock:
        if user_id in _in_flight:
            return False
        _in_flight.add(user_id)
        return True


async def _release(user_id: str) -> None:
    async with _in_flight_lock:
        _in_flight.discard(user_id)


async def run_agent_for_user(user_id: str, trigger_reason: str) -> None:
    if not await _claim(user_id):
        logger.info(
            "Agent run skipped for user=%s reason=%s — an agent run is already in flight",
            user_id, trigger_reason,
        )
        return
    try:
        logger.info("Agent run starting for user=%s reason=%s", user_id, trigger_reason)
        await agent_graph.ainvoke({"user_id": user_id, "trigger_reason": trigger_reason})
        logger.info("Agent run completed for user=%s", user_id)
    except Exception:
        logger.exception("Agent run failed for user=%s", user_id)
    finally:
        await _release(user_id)