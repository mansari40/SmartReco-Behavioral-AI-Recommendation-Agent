"""Shared FastAPI dependencies: current user + admin-only guard."""
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.security import decode_access_token
from app.db.session import get_db
from app.models.user import User, UserRole

# Kept for OpenAPI docs (Authorize button). Actual extraction is manual so
# browser page navigations can authenticate via cookie.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

COOKIE_NAME = "upulse_token"


def _extract_token(request: Request) -> str | None:
    """Accept the JWT from the Authorization header (API calls) or the
    upulse_token cookie (server-side-protected pages like /admin, which
    are reached by browser navigation and cannot send headers)."""
    header = request.headers.get("Authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.cookies.get(COOKIE_NAME)


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    token = _extract_token(request)
    user_id = decode_access_token(token) if token else None
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user