"""
pgvector-backed vector store for PostgreSQL — the production vector store.

Implements the same async API as the Chroma dev/test fallback and returns
Chroma-shaped query results, so callers cannot tell which backend is active:

    upsert_product(vector_id, embedding, document, metadata)
    delete_product(vector_id)
    product_exists(vector_id) -> bool
    query(embedding, top_k=10, where=None)
        -> {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}

Semantic ranking uses pgvector's cosine distance (<->) over
course_embeddings.embedding. Only imported when DATABASE_URL is PostgreSQL.
"""
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.session import AsyncSessionLocal
from app.models.course_embedding import CourseEmbedding


def _metadata_filter(where: dict | None) -> list:
    """Translate Chroma-style where dicts into JSONB predicates. Supports the
    equality and $in forms used by the retrieve node."""
    conditions = []
    if not where:
        return conditions
    for key, value in where.items():
        column = CourseEmbedding.metadata_[key].astext
        if isinstance(value, dict):
            if "$in" in value:
                conditions.append(column.in_([str(v) for v in value["$in"]]))
            elif "$eq" in value:
                conditions.append(column == str(value["$eq"]))
        else:
            conditions.append(column == str(value))
    return conditions


async def upsert_product(vector_id: str, embedding: list[float], document: str, metadata: dict) -> None:
    stmt = pg_insert(CourseEmbedding).values(
        vector_id=vector_id,
        document=document,
        metadata_=metadata,
        embedding=embedding,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[CourseEmbedding.vector_id],
        set_={
            "document": stmt.excluded.document,
            "metadata": stmt.excluded.metadata,  # set_ keys are DB column names
            "embedding": stmt.excluded.embedding,
            "updated_at": datetime.now(timezone.utc),
        },
    )
    async with AsyncSessionLocal() as db:
        await db.execute(stmt)
        await db.commit()


async def delete_product(vector_id: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            delete(CourseEmbedding).where(CourseEmbedding.vector_id == vector_id)
        )
        await db.commit()


async def product_exists(vector_id: str) -> bool:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(CourseEmbedding.vector_id).where(CourseEmbedding.vector_id == vector_id)
        )
        return result.scalar_one_or_none() is not None


async def query(embedding: list[float], top_k: int = 10, where: dict | None = None) -> dict:
    distance = CourseEmbedding.embedding.cosine_distance(embedding)
    stmt = select(CourseEmbedding, distance.label("distance"))
    conditions = _metadata_filter(where)
    if conditions:
        stmt = stmt.where(*conditions)
    stmt = stmt.order_by(distance).limit(top_k)

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(stmt)).all()

    return {
        "ids": [[r[0].vector_id for r in rows]],
        "documents": [[r[0].document for r in rows]],
        "metadatas": [[r[0].metadata_ for r in rows]],
        "distances": [[r[1] for r in rows]],
    }
