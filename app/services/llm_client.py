"""
Single choke point for ALL LLM/embedding calls in the app. Nothing else in
the codebase should import the OpenAI SDK directly — go through this
module so every AI call is auditable in one place. Every real call
(success or failure) is logged to LLMCallLog — that log is the source of
truth for the Agent Console; nothing there is invented.
"""
import hashlib
import logging
import random
import time

from openai import AsyncOpenAI, APIStatusError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.db.session import AsyncSessionLocal
from app.models.llm_call_log import LLMCallLog

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

# Only retry errors that are plausibly transient. A 400/404 (bad model
# name, malformed request) will fail identically on every retry — retrying
# those just burns quota and time for a guaranteed-to-fail call. 429 (rate
# limit) and 5xx (server-side) are worth retrying; everything else isn't.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return True  # network errors, timeouts, etc. — worth retrying


async def _log_call(
    call_type: str,
    model: str,
    latency_ms: int,
    success: bool,
    is_mock: bool = False,
    usage: dict | None = None,
    error: str | None = None,
) -> None:
    """Best-effort logging — a failure here must never break the actual
    LLM call it's describing, so any exception is swallowed."""
    try:
        async with AsyncSessionLocal() as db:
            db.add(LLMCallLog(
                call_type=call_type,
                model=model,
                is_mock=is_mock,
                latency_ms=latency_ms,
                prompt_tokens=(usage or {}).get("prompt_tokens"),
                completion_tokens=(usage or {}).get("completion_tokens"),
                total_tokens=(usage or {}).get("total_tokens"),
                success=success,
                error=error,
            ))
            await db.commit()
    except Exception:
        pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
)
async def chat_completion(
    messages: list[dict],
    model: str | None = None,
    response_format_json: bool = False,
    temperature: float = 0.7,
) -> str:
    """Returns raw text content. Callers that need structured output should
    parse defensively — not every model honors OpenAI's response_format
    param, so we ask for JSON in-prompt too, elsewhere."""
    resolved_model = model or settings.llm_chat_model

    if response_format_json:
        try:
            return await _chat_once(messages, resolved_model, temperature, json_mode=True)
        except _EmptyChoicesError:
            # Some providers silently ignore response_format and return 200
            # with an empty body (e.g. deepseek-v3 via OpenRouter). The
            # prompts already ask for JSON in-prompt and every caller parses
            # defensively, so retry once without the param instead of
            # failing the whole agent run.
            logger.warning("Empty response with json mode for %s — retrying without response_format", resolved_model)
            return await _chat_once(messages, resolved_model, temperature, json_mode=False)
    return await _chat_once(messages, resolved_model, temperature, json_mode=False)


class _EmptyChoicesError(RuntimeError):
    pass


async def _chat_once(messages: list[dict], model: str, temperature: float, json_mode: bool) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.monotonic()
    try:
        response = await _client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            # OpenRouter occasionally returns 200 with an empty body —
            # treat as a transient failure so the retry wrapper handles it.
            raise _EmptyChoicesError("LLM response contained no choices")
        latency_ms = int((time.monotonic() - start) * 1000)
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        await _log_call("chat", model, latency_ms, success=True, usage=usage)
        return response.choices[0].message.content or ""
    except _EmptyChoicesError:
        latency_ms = int((time.monotonic() - start) * 1000)
        await _log_call("chat", model, latency_ms, success=False, error="empty response body")
        raise
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        await _log_call("chat", model, latency_ms, success=False, error=str(exc)[:500])
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
)
async def get_embedding(text: str, model: str | None = None) -> list[float]:
    """DEV-ONLY ESCAPE HATCH: if settings.mock_embeddings is True, returns a
    deterministic fake vector instead of calling out — logged honestly as
    is_mock=True so the console never conflates it with a real call."""
    resolved_model = model or settings.llm_embedding_model

    if settings.mock_embeddings:
        start = time.monotonic()
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        vector = [rng.uniform(-1, 1) for _ in range(768)]
        latency_ms = int((time.monotonic() - start) * 1000)
        await _log_call("embedding", "mock", latency_ms, success=True, is_mock=True)
        return vector

    start = time.monotonic()
    try:
        response = await _client.embeddings.create(model=resolved_model, input=text)
        latency_ms = int((time.monotonic() - start) * 1000)
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        await _log_call("embedding", resolved_model, latency_ms, success=True, usage=usage)
        return response.data[0].embedding
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        await _log_call("embedding", resolved_model, latency_ms, success=False, error=str(exc)[:500])
        raise