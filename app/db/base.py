"""
Declarative base + a single import point that pulls in every model so
Base.metadata is complete when init_db() runs create_all().
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import models so they register with Base.metadata. Kept at the bottom of
# the file to avoid circular imports between the models and Base itself.
from app.models import user, product, event, cognitive_model, recommendation  # noqa: E402,F401


async def init_db() -> None:
    """Create tables if they don't exist. Fine for a hackathon timeline;
    swap for Alembic migrations if the project outlives the demo."""
    from app.db.session import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)