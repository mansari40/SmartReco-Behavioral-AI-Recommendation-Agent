"""
Vector-store facade.

The backend is chosen once at import time from settings.database_url:

    PostgreSQL -> pgvector-backed store (app.services.vector_store_pg)
    SQLite     -> local Chroma fallback (app.services.vector_store_chroma),
                  used only for local dev and the offline pytest suite.

Both backends expose the same async API and return Chroma-shaped query
results, so callers (products router, retrieve node, seeder) never need to
know which backend is active.
"""
from app.config import settings

if settings.database_url.startswith("postgres"):
    from app.services.vector_store_pg import delete_product, product_exists, query, upsert_product  # noqa: F401

    BACKEND = "pgvector"
else:
    from app.services.vector_store_chroma import delete_product, product_exists, query, upsert_product  # noqa: F401

    BACKEND = "chroma"
