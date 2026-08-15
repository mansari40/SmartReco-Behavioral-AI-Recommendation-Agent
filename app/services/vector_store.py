"""
Vector-store facade.

The backend is selected lazily, on the first call, from
settings.database_url — importing this module never pulls Chroma (or its
onnxruntime dependency) into the process:

    PostgreSQL -> pgvector-backed store (app.services.vector_store_pg)
    SQLite     -> local Chroma fallback (app.services.vector_store_chroma),
                  used only for local dev and the offline pytest suite.

Both backends expose the same async API and return Chroma-shaped query
results, so callers (products router, retrieve node, seeder) never need to
know which backend is active.
"""
from functools import lru_cache

from app.config import settings


@lru_cache
def _get_backend():
    """Return the active backend module, importing it only on first use —
    never at import time. This keeps Chroma off the production (PostgreSQL)
    import graph entirely."""
    if settings.database_url.startswith("postgres"):
        from app.services import vector_store_pg as backend
    else:
        from app.services import vector_store_chroma as backend
    return backend


async def upsert_product(vector_id: str, embedding: list[float], document: str, metadata: dict) -> None:
    await _get_backend().upsert_product(vector_id, embedding, document, metadata)


async def delete_product(vector_id: str) -> None:
    await _get_backend().delete_product(vector_id)


async def product_exists(vector_id: str) -> bool:
    return await _get_backend().product_exists(vector_id)


async def query(embedding: list[float], top_k: int = 10, where: dict | None = None) -> dict:
    return await _get_backend().query(embedding, top_k, where)