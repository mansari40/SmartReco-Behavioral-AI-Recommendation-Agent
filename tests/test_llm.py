"""LLM client resilience: the OpenRouter gateway sometimes returns 200 with
an empty body; json-mode calls must fall back to a non-json retry instead of
failing the agent run. Uses a stub OpenAI client — no network."""
import pytest

from app.services import llm_client
from tests.conftest import ORIGINAL_CHAT_COMPLETION


class _OkMessage:
    content = "hello from stub"


class EmptyChoicesStub:
    choices = []


class OkStub:
    class Choice:
        def __init__(self):
            self.message = _OkMessage()
            self.finish_reason = "stop"

    choices = [Choice()]

    class Usage:
        prompt_tokens = 5
        completion_tokens = 3
        total_tokens = 8

    usage = Usage()


@pytest.mark.asyncio
async def test_json_mode_falls_back_when_provider_returns_empty(monkeypatch):
    """First call (json mode) returns empty choices → must retry without
    response_format and still succeed."""
    calls = []

    async def fake_create(model, messages, temperature, **kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            return EmptyChoicesStub()
        return OkStub()

    monkeypatch.setattr(llm_client, "chat_completion", ORIGINAL_CHAT_COMPLETION)
    monkeypatch.setattr(llm_client._client.chat.completions, "create", fake_create)

    reply = await llm_client.chat_completion(
        messages=[{"role": "user", "content": "hi"}], response_format_json=True
    )
    assert reply == "hello from stub"
    assert len(calls) == 2
    assert "response_format" in calls[0]  # first attempt asked for JSON
    assert "response_format" not in calls[1]  # fallback dropped it


@pytest.mark.asyncio
async def test_plain_call_passes_through(monkeypatch):
    async def fake_create(model, messages, temperature, **kwargs):
        return OkStub()

    monkeypatch.setattr(llm_client, "chat_completion", ORIGINAL_CHAT_COMPLETION)
    monkeypatch.setattr(llm_client._client.chat.completions, "create", fake_create)

    reply = await llm_client.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert reply == "hello from stub"