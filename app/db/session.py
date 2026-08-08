"""
Async SQLAlchemy engine/session setup. Swapping DATABASE_URL to a Postgres
async URL (postgresql+asyncpg://...)
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False, future=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields a request-scoped async session."""
    async with AsyncSessionLocal() as session:
        yield session