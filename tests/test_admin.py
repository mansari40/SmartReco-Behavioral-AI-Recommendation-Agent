"""Admin console tests: the read-only Overview / Users / Observability API
and its server-rendered pages. Verifies role gating (admin only) and that
the reported numbers trace back to real seeded rows."""
import uuid
from datetime import datetime, timedelta, timezone

from app.db.session import AsyncSessionLocal
from app.models.cognitive_model import DecisionStage, PriceSensitivity, UserCognitiveModel
from app.models.event import Event, EventType
from app.models.llm_call_log import LLMCallLog
from app.models.product import Product, SyncStatus
from app.models.recommendation import Recommendation
from tests.conftest import auth


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _insert_product(db, title="Test Course", sync=SyncStatus.SYNCED, error=None) -> Product:
    product = Product(
        title=title,
        description="desc",
        category="AI",
        price=49.99,
        vector_id=str(uuid.uuid4()),
        sync_status=sync,
        sync_error=error,
    )
    db.add(product)
    await db.flush()
    return product


async def _user_id(client, token: str) -> str:
    return (await client.get("/api/auth/me", headers=auth(token))).json()["id"]


async def test_admin_api_forbidden_for_regular_user(client, user_token):
    for path in ("/api/admin/overview", "/api/admin/users", "/api/admin/observability"):
        resp = await client.get(path, headers=auth(user_token))
        assert resp.status_code == 403


async def test_admin_api_requires_token(client):
    for path in ("/api/admin/overview", "/api/admin/users", "/api/admin/observability"):
        assert (await client.get(path)).status_code == 401


async def test_admin_user_detail_404_and_gated(client, admin_token, user_token):
    resp = await client.get("/api/admin/users/does-not-exist", headers=auth(admin_token))
    assert resp.status_code == 404
    resp = await client.get("/api/admin/users/does-not-exist", headers=auth(user_token))
    assert resp.status_code == 403


async def test_overview_aggregates_real_rows(client, admin_token, user_token):
    uid = await _user_id(client, user_token)
    async with AsyncSessionLocal() as db:
        ok = await _insert_product(db)
        await _insert_product(db, title="Broken Course", sync=SyncStatus.FAILED, error="embedding failed")
        for _ in range(3):
            db.add(Event(user_id=uid, event_type=EventType.PRODUCT_VIEW, product_id=ok.id, event_metadata={}))
        db.add(UserCognitiveModel(
            user_id=uid,
            stated_intents=["llm"],
            inferred_intents=["AI agents"],
            decision_stage=DecisionStage.INTEREST,
            purchase_readiness=0.6,
            price_sensitivity=PriceSensitivity.MEDIUM,
            category_affinity=["AI"],
        ))
        for _ in range(2):
            db.add(LLMCallLog(
                call_type="chat", model="test-model", is_mock=False, latency_ms=120,
                prompt_tokens=100, completion_tokens=50, total_tokens=150, success=True,
            ))
        await db.commit()

    resp = await client.get("/api/admin/overview", headers=auth(admin_token))
    assert resp.status_code == 200
    d = resp.json()

    assert d["total_users"] >= 2
    assert d["active_users_24h"] >= 1
    assert d["model_calls_24h"] >= 2
    assert d["real_calls_24h"] >= 2

    assert d["vector_sync"]["total"] == 2
    assert d["vector_sync"]["synced"] == 1
    assert d["vector_sync"]["failed"] == 1
    assert d["vector_sync"]["sample_errors"][0]["error"] == "embedding failed"

    assert d["decision_stage_distribution"].get("interest") == 1
    assert d["price_sensitivity_distribution"].get("medium") == 1
    assert d["avg_purchase_readiness"] == 0.6
    assert any(i["name"] == "AI agents" for i in d["top_inferred_intents"])

    assert d["agent_runs"]["runs_24h"] >= 1
    assert d["cost"]["total_tokens"] >= 300
    assert d["cost"]["estimated_usd"] > 0


async def test_users_list_and_detail(client, admin_token, user_token):
    uid = await _user_id(client, user_token)
    me = (await client.get("/api/auth/me", headers=auth(user_token))).json()
    async with AsyncSessionLocal() as db:
        db.add(UserCognitiveModel(
            user_id=uid,
            stated_intents=[],
            inferred_intents=["cloud"],
            decision_stage=DecisionStage.AWARENESS,
            purchase_readiness=0.2,
            price_sensitivity=PriceSensitivity.HIGH,
            category_affinity=[],
        ))
        db.add(Recommendation(
            user_id=uid,
            narrative="n",
            products=[{"product_id": "x", "reason": "r"}],
            persuasion_strategy="scarcity",
            confidence=0.9,
            trigger_reason="test trigger",
        ))
        await db.commit()

    resp = await client.get("/api/admin/users", headers=auth(admin_token))
    assert resp.status_code == 200
    page = resp.json()
    assert page["total"] >= 2

    mine = next((u for u in page["items"] if u["email"] == me["email"]), None)
    assert mine is not None
    assert mine["events_count"] == 0
    assert mine["recommendations_count"] == 1
    assert mine["decision_stage"] == "awareness"
    assert mine["price_sensitivity"] == "high"
    assert mine["purchase_readiness"] == 0.2
    assert mine["has_active_recommendation"] is True

    detail = await client.get("/api/admin/users/" + mine["id"], headers=auth(admin_token))
    assert detail.status_code == 200
    d = detail.json()
    assert d["email"] == me["email"]
    assert d["cognitive_profile"]["decision_stage"] == "awareness"
    assert d["cognitive_profile"]["inferred_intents"] == ["cloud"]
    assert len(d["recommendations"]) == 1
    assert d["recommendations"][0]["product_count"] == 1
    assert d["recommendations"][0]["trigger_reason"] == "test trigger"


async def test_observability_groups_calls_into_runs(client, admin_token, user_token):
    uid = await _user_id(client, user_token)
    me = (await client.get("/api/auth/me", headers=auth(user_token))).json()
    base = _now()
    async with AsyncSessionLocal() as db:
        # Run 1: two chat calls + one embedding call, ~5 minutes ago.
        for offset in (300, 299, 298):
            t = base - timedelta(seconds=offset)
            if offset == 298:
                db.add(LLMCallLog(
                    call_type="embedding", model="embed-model", is_mock=False, latency_ms=30,
                    total_tokens=1000, success=True, created_at=t,
                ))
            else:
                db.add(LLMCallLog(
                    call_type="chat", model="test-model", is_mock=False, latency_ms=120,
                    prompt_tokens=100, completion_tokens=50, total_tokens=150, success=True, created_at=t,
                ))
        # Run 2: one chat call 60s ago — 238s gap separates it from run 1.
        db.add(LLMCallLog(
            call_type="chat", model="test-model", is_mock=False, latency_ms=120,
            prompt_tokens=100, completion_tokens=50, total_tokens=150, success=True,
            created_at=base - timedelta(seconds=60),
        ))
        # Recommendation written by run 2's store node right after its last call.
        db.add(Recommendation(
            user_id=uid,
            narrative="n",
            products=[{"product_id": "x", "reason": "r"}],
            confidence=0.8,
            trigger_reason="trigger for run 2",
            persuasion_strategy="social proof",
            created_at=base - timedelta(seconds=55),
        ))
        await db.commit()

    resp = await client.get("/api/admin/observability", headers=auth(admin_token))
    assert resp.status_code == 200
    d = resp.json()

    assert len(d["runs"]) == 2
    assert d["calls_24h"] == 4
    assert d["total_tokens_24h"] == 1450
    assert abs(d["estimated_cost_usd_24h"] - 0.00029) < 0.00001

    run1, run2 = d["runs"][0], d["runs"][1]
    assert run1["llm_calls"] == 3
    assert run1["chat_calls"] == 2
    assert run1["embedding_calls"] == 1
    assert run1["success"] is True
    assert run2["llm_calls"] == 1

    matched = [r for r in d["runs"] if r["recommendation"]]
    assert len(matched) == 1
    m = matched[0]
    assert m["recommendation"]["user_email"] == me["email"]
    assert m["recommendation"]["trigger_reason"] == "trigger for run 2"
    assert m["recommendation"]["persuasion_strategy"] == "social proof"
    assert m["llm_calls"] == 1


async def test_admin_pages_role_gated(client, user_token, admin_token):
    for path in ("/admin/overview", "/admin/users", "/admin/observability"):
        resp = await client.get(path, headers=auth(user_token))
        assert resp.status_code == 403
        resp = await client.get(path, headers=auth(admin_token))
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")


async def test_admin_user_detail_page_role_gated(client, user_token, admin_token):
    resp = await client.get("/admin/users/whatever", headers=auth(user_token))
    assert resp.status_code == 403
    resp = await client.get("/admin/users/whatever", headers=auth(admin_token))
    assert resp.status_code == 200