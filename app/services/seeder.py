"""
Idempotent startup bootstrap: the single admin account plus the canonical
52-course catalog (with vector embeddings). Safe to run on every startup —
creates only what's missing, never duplicates courses, embeddings, or the
admin.

This replaces the old manual ceremony (create account -> promote to admin ->
run scripts/seed_products.py) that was previously needed after every Render
sleep/restart.
"""
import logging
import uuid

from sqlalchemy import select

from app.config import settings
from app.db.session import AsyncSessionLocal
from app.models.product import Product, SyncStatus
from app.models.user import User, UserRole
from app.security import hash_password
from app.services import llm_client, vector_store
from app.services.catalog_meta import infer_level, seed_rating
from scripts.seed_products import CATALOG

logger = logging.getLogger(__name__)


def _stable_vector_id(title: str) -> str:
    """Deterministic vector id derived from the course title, so restarts
    never produce a second vector entry for the same course."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"upulse-course/{title}"))


async def sync_product_to_vector_store(product: Product) -> tuple[SyncStatus, str | None]:
    """Embed a product and upsert it into the active vector store. Returns
    the dual-write grade (SYNCED/FAILED + error) the rest of the app uses."""
    try:
        embedding = await llm_client.get_embedding(product.to_embedding_text())
        await vector_store.upsert_product(
            vector_id=product.vector_id,
            embedding=embedding,
            document=product.to_embedding_text(),
            metadata={
                "category": product.category,
                "price": product.price,
                "sql_id": product.id,
                "level": product.level,
                "rating": product.rating,
            },
        )
        return SyncStatus.SYNCED, None
    except Exception as exc:  # noqa: BLE001 — any failure must surface, not vanish
        return SyncStatus.FAILED, str(exc)


async def ensure_admin() -> None:
    """Create the single intended admin (ADMIN_EMAIL / ADMIN_PASSWORD) if and
    only if it doesn't exist yet. Idempotent: an existing account with that
    email is left untouched, including its role."""
    email = settings.admin_email
    password = settings.admin_password
    if not email or not password:
        logger.warning("ADMIN_EMAIL/ADMIN_PASSWORD not configured — skipping admin bootstrap")
        return

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user:
            if user.role != UserRole.ADMIN:
                user.role = UserRole.ADMIN
                await db.commit()
                logger.info("Promoted existing account %s to admin", email)
            else:
                logger.info("Admin %s already exists — leaving as is", email)
            return
        db.add(User(email=email, hashed_password=hash_password(password), role=UserRole.ADMIN))
        await db.commit()
        logger.info("Created admin account %s", email)


async def seed_catalog() -> dict:
    """Ensure the canonical 52 courses exist with embeddings. Idempotent:
    existing courses are left untouched, and a course is embedded only if
    its vector is actually missing from the active vector store — an
    existing embedding is reused on every subsequent boot (no re-embed cost).
    sync_status alone is not trusted: a course marked PENDING but already
    present in the store (e.g. after a store migration) is not re-embedded."""
    stats = {"total": len(CATALOG), "created": 0, "embedded": 0, "skipped": 0, "failed": 0}

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Product.title))
        existing_titles = {row[0] for row in result.all()}

        for item in CATALOG:
            if item["title"] in existing_titles:
                product = (
                    await db.execute(select(Product).where(Product.title == item["title"]))
                ).scalar_one()
            else:
                product = Product(
                    title=item["title"],
                    description=item["description"],
                    category=item["category"],
                    price=item["price"],
                    vector_id=_stable_vector_id(item["title"]),
                    sync_status=SyncStatus.PENDING,
                )
                db.add(product)
                await db.flush()
                existing_titles.add(item["title"])
                stats["created"] += 1

            if product.level is None or product.rating is None:
                product.level = infer_level(product.title)
                product.rating, product.rating_count = seed_rating(product.id)

            if await vector_store.product_exists(product.vector_id):
                product.sync_status = SyncStatus.SYNCED
                product.sync_error = None
                stats["skipped"] += 1
                continue

            product.sync_status, product.sync_error = await sync_product_to_vector_store(product)
            if product.sync_status == SyncStatus.SYNCED:
                stats["embedded"] += 1
            else:
                stats["failed"] += 1
                logger.error("Failed to embed course %r: %s", product.title, product.sync_error)

        await db.commit()

    return stats
