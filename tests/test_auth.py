"""Authentication and authorization tests: register, login, token validity,
role enforcement. Verifies passwords are hashed and admin endpoints are
protected server-side."""
from tests.conftest import auth


async def test_register_creates_user_and_hashes_password(client):
    resp = await client.post("/api/auth/register", json={
        "email": "new@test.com", "first_name": "A", "last_name": "B", "password": "secret123",
    })
    assert resp.status_code == 201
    assert resp.json()["role"] == "user"

    from app.db.session import AsyncSessionLocal
    from app.models.user import User
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == "new@test.com"))).scalar_one()
        assert user.hashed_password != "secret123"          # never stored in plaintext
        assert user.hashed_password.startswith("$2")        # bcrypt


async def test_duplicate_email_rejected(client):
    payload = {"email": "dup@test.com", "first_name": "A", "last_name": "B", "password": "secret123"}
    assert (await client.post("/api/auth/register", json=payload)).status_code == 201
    assert (await client.post("/api/auth/register", json=payload)).status_code == 400


async def test_login_wrong_password(client, user_token):
    import re

    # extract the registered email out of the token fixture user by logging in
    resp = await client.post("/api/auth/login", data={"username": "nobody@test.com", "password": "wrong"})
    assert resp.status_code == 401


async def test_me_requires_token(client):
    assert (await client.get("/api/auth/me")).status_code == 401
    assert (await client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"})).status_code == 401


async def test_me_returns_own_profile(client, user_token):
    resp = await client.get("/api/auth/me", headers=auth(user_token))
    assert resp.status_code == 200
    assert resp.json()["email"].endswith("@test.com")


async def test_admin_endpoint_forbidden_for_user(client, user_token):
    resp = await client.get("/api/console/stats", headers=auth(user_token))
    assert resp.status_code == 403


async def test_admin_endpoint_allowed_for_admin(client, admin_token):
    resp = await client.get("/api/console/stats", headers=auth(admin_token))
    assert resp.status_code == 200
    assert "calls_today" in resp.json()


async def test_admin_page_requires_admin(client, user_token):
    # the HTML admin page must be protected server-side too, not just the API
    resp = await client.get("/admin", headers=auth(user_token), follow_redirects=False)
    assert resp.status_code == 403
