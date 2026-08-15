from fastapi import APIRouter, Depends, HTTPException, status
import asyncio
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


from app.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models.user import User, UserRole
from app.routers.deps import get_current_user
from app.schemas.user import Token, UserCreate, UserOut
from app.services.email_service import send_password_reset_email
from app.config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")

    user = User(
        email=payload.email,
        first_name=payload.first_name,
        last_name=payload.last_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return Token(access_token=create_access_token(subject=user.id))


@router.post("/password-reset", status_code=status.HTTP_202_ACCEPTED)
async def request_password_reset(payload: dict[str, str], db: AsyncSession = Depends(get_db)):
    email = payload.get("email")
    if not email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email required")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user:
        await asyncio.to_thread(send_password_reset_email, user)
    return {"status": "ok"}


@router.post("/password-reset/confirm", status_code=status.HTTP_200_OK)
async def reset_password(payload: dict[str, str], db: AsyncSession = Depends(get_db)):
    token = payload.get("token")
    password = payload.get("password")
    if not token or not password:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Token and password required")

    from app.security import decode_password_reset_token

    user_id = decode_password_reset_token(token)
    if not user_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired reset token")
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    user.hashed_password = hash_password(password)
    await db.commit()
    return {"status": "ok"}


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user


@router.post("/promote-to-admin", status_code=status.HTTP_200_OK)
async def promote_to_admin(payload: dict[str, str], db: AsyncSession = Depends(get_db)):
    """TEMPORARY, ONE-TIME-USE endpoint to bootstrap the first admin account
    on a fresh deploy where no admin exists yet and there's no DB shell
    access. Protected by a secret (not a real auth token) since this
    bypasses normal permission checks. DELETE THIS after using it once."""
    email = payload.get("email")
    secret = payload.get("secret")

    if not settings.admin_promote_secret or secret != settings.admin_promote_secret:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid secret")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    user.role = UserRole.ADMIN
    await db.commit()
    return {"status": "ok", "email": user.email, "role": user.role.value}