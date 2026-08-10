"""
Single choke point for ALL LLM/embedding calls in the app. Nothing else in
the codebase should import the OpenAI SDK directly — go through this module
so every AI call is auditable in one place, and so retries live in exactly
one spot. Provider-agnostic by design: currently pointed at OpenRouter via
config, swappable to any OpenAI-compatible gateway by changing .env only.
"""
import hashlib
import random

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

_client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def chat_completion(
    messages: list[dict],
    model: str | None = None,
    response_format_json: bool = False,
    temperature: float = 0.7,
) -> str:
    """Returns raw text content. Callers that need structured output should
    parse defensively — not every model honors OpenAI's response_format
    param, so we ask for JSON in-prompt too, elsewhere."""
    kwargs = {}
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = await _client.chat.completions.create(
        model=model or settings.llm_chat_model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def get_embedding(text: str, model: str | None = None) -> list[float]:
    """DEV-ONLY ESCAPE HATCH: if settings.mock_embeddings is True, returns a
    deterministic fake vector instead of calling out. Must be False for any
    real run — exists so local dev isn't blocked by provider issues."""
    if settings.mock_embeddings:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.uniform(-1, 1) for _ in range(768)]

    response = await _client.embeddings.create(
        model=model or settings.llm_embedding_model,
        input=text,
    )
    return response.data[0].embedding