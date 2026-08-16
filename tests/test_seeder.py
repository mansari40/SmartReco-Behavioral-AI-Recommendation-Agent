"""
Regression: seed_catalog() must be idempotent even across repeated runs and
even when legacy duplicate rows exist (the MultipleResultsFound crash seen
in smartreco.db: 58 rows for 52 canonical titles).
"""
import uuid

from sqlalchemy import func, select, text

from app.db.session import AsyncSessionLocal
from app.models.product import Product, SyncStatus
from app.services.seeder import seed_catalog


async def test_seed_catalog_twice_does_not_grow():
    first = await seed_catalog()
    assert first["total"] == 52
    assert first["created"] == 52

    second = await seed_catalog()
    assert second["created"] == 0

    async with AsyncSessionLocal() as db:
        count = (await db.execute(select(func.count()).select_from(Product))).scalar_one()
    assert count == 52


async def test_seed_catalog_tolerates_legacy_duplicate_titles():
    async with AsyncSessionLocal() as db:
        await db.execute(text("DROP INDEX IF EXISTS uq_products_title"))
        db.add_all([
            Product(
                title="Agentic AI Fundamentals", description="legacy dup a", category="AI",
                price=10.0, vector_id=str(uuid.uuid4()), sync_status=SyncStatus.PENDING,
            ),
            Product(
                title="Agentic AI Fundamentals", description="legacy dup b", category="AI",
                price=20.0, vector_id=str(uuid.uuid4()), sync_status=SyncStatus.PENDING,
            ),
        ])
        await db.commit()

    try:
        stats = await seed_catalog()

        assert stats["total"] == 52
        assert stats["created"] == 51

        async with AsyncSessionLocal() as db:
            dup_count = (
                await db.execute(
                    select(func.count()).select_from(Product)
                    .where(Product.title == "Agentic AI Fundamentals")
                )
            ).scalar_one()
        assert dup_count == 2
    finally:
        async with AsyncSessionLocal() as db:
            await db.execute(text("DELETE FROM products WHERE title = 'Agentic AI Fundamentals'"))
            await db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_products_title ON products (title)"))
            await db.commit()