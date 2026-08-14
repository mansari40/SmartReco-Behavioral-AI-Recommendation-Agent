"""
Pytest fixtures for the SMARTreco test suite.

Environment is redirected to isolated test resources (temp SQLite DB,
temp Chroma dir) BEFORE any app module is imported, so tests never touch
the development database or vector store. All LLM/embedding calls are
mocked deterministically — the suite runs fully offline.
"""
import hashlib
import os
import tempfile
import uuid

TEST_DIR = tempfile.mkdtemp(prefix="smartreco-test-")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DIR}/test.db"
os.environ["CHROMA_PERSIST_DIR"] = f"{TEST_DIR}/chroma"
os.environ["MOCK_EMBEDDINGS"] = "true"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["LLM_CHAT_MODEL"] = "test-model"

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.db.base import init_db  # noqa: E402
from app.db.session import AsyncSessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.services import llm_client  # noqa: E402

EMBEDDING_DIM = 8

# Captured before the autouse fake_llm fixture patches the module — lets
# tests exercise the real retry/fallback logic with a stubbed HTTP client.
ORIGINAL_CHAT_COMPLETION = llm_client.chat_completion
ORIGINAL_GET_EMBEDDING = llm_client.get_embedding


def fake_embedding(text: str) -> list[float]:
    """Deterministic small vector for offline embedding tests."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    out = []
    for i in range(EMBEDDING_DIM):
        seed = (seed * 1103515245 + 12345) % (2**31)
        out.append((seed % 2000) / 1000.0 - 1.0)
    return out


class FakeLLM:
    """Canned LLM responses, dispatched on prompt content so each node gets
    a sensible reply. Tests may override via `fake_llm.responses`."""

    def __init__(self):
        self.responses = {}
        self.chat_calls = 0
        self.embedding_calls = 0

    async def chat_completion(self, messages, model=None, response_format_json=False, temperature=0.7):
        self.chat_calls += 1
        prompt = "\n".join(m["content"] for m in messages)
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return self.default_response(prompt)

    async def get_embedding(self, text, model=None):
        self.embedding_calls += 1
        return fake_embedding(text)

    @staticmethod
    def default_response(prompt: str) -> str:
        import json
        import re

        if "user-behavior analyst" in prompt or "CURRENT PROFILE" in prompt:
            return json.dumps({
                "stated_intents": ["agentic ai"],
                "inferred_intents": ["building AI agents"],
                "decision_stage": "interest",
                "purchase_readiness": 0.5,
                "price_sensitivity": "medium",
                "detected_objections": [],
                "brand_affinity": [],
                "category_affinity": ["AI"],
                "session_arc": "User explored agentic AI and LangGraph.",
            })
        if "evaluating product candidates" in prompt or "CANDIDATES" in prompt:
            ids = re.findall(r"product_id: ([\w-]+)", prompt)
            return json.dumps({"scores": [
                {"product_id": pid, "relevance_score": 0.9, "reasoning": "matches interests"} for pid in ids
            ]})
        if "writing a personalized product recommendation" in prompt:
            ids = re.findall(r"product_id: ([\w-]+)", prompt)
            return json.dumps({
                "narrative": "Based on your recent activity, these courses build directly on your interests.",
                "product_reasons": [{"product_id": pid, "reason": "builds on what you explored"} for pid in ids],
                "reasoning_chain": ["user showed interest in agentic AI"],
            })
        if "critiquing a generated product recommendation" in prompt:
            return json.dumps({"should_regenerate": False, "feedback": "", "confidence": 0.8})
        return '{"ok": true}'


@pytest_asyncio.fixture(autouse=True)
async def _isolated_db():
    """Fresh empty tables for every test."""
    from sqlalchemy import text

    async with AsyncSessionLocal() as db:
        for table in ("events", "recommendations", "user_cognitive_models",
                      "llm_call_logs", "products", "users"):
            await db.execute(text(f"DELETE FROM {table}"))
        await db.commit()
    yield


@pytest_asyncio.fixture(autouse=True)
def fake_llm(monkeypatch):
    """Replace real LLM/embedding calls with the deterministic fake."""
    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "chat_completion", fake.chat_completion)
    monkeypatch.setattr(llm_client, "get_embedding", fake.get_embedding)
    return fake


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _init_app_db():
    await init_db()
    yield


@pytest_asyncio.fixture
async def client():
    """Async HTTP client against the running ASGI app (no lifespan — DB is
    initialized by the session fixture; the scheduler is not started)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def user_token(client) -> str:
    """Registered, logged-in regular user."""
    email = f"user-{uuid.uuid4().hex[:8]}@test.com"
    resp = await client.post("/api/auth/register", json={
        "email": email, "first_name": "Test", "last_name": "User", "password": "testpass123",
    })
    assert resp.status_code == 201
    resp = await client.post("/api/auth/login", data={"username": email, "password": "testpass123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest_asyncio.fixture
async def admin_token(client, fake_llm) -> str:
    """Admin user with role upgraded directly in the DB."""
    from app.models.user import User, UserRole
    from app.security import hash_password

    email = f"admin-{uuid.uuid4().hex[:8]}@test.com"
    async with AsyncSessionLocal() as db:
        db.add(User(email=email, hashed_password=hash_password("adminpass123"), role=UserRole.ADMIN))
        await db.commit()
    resp = await client.post("/api/auth/login", data={"username": email, "password": "adminpass123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
