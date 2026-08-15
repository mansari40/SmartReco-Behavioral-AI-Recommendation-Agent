"""
Declarative base + a single import point that pulls in every model so
Base.metadata is complete when init_db() runs create_all().
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import models so they register with Base.metadata. Kept at the bottom of
# the file to avoid circular imports between the models and Base itself.
from app.models import user, product, event, cognitive_model, recommendation, llm_call_log, course_embedding  # noqa: E402,F401


def _backfill_product_meta(sync_conn) -> None:
    """Course level + seeded rating/rating-count for products that predate
    the columns. Deterministic: level from title keywords, rating from a
    stable hash of the product id. Dialect-agnostic (named bind params)."""
    from sqlalchemy import text

    from app.services.catalog_meta import infer_level, seed_rating

    for product_id, title in sync_conn.exec_driver_sql(
        "SELECT id, title FROM products WHERE level IS NULL OR rating IS NULL"
    ).fetchall():
        rating, count = seed_rating(product_id)
        sync_conn.execute(
            text(
                "UPDATE products SET level = :level, rating = :rating, "
                "rating_count = :count WHERE id = :id"
            ),
            {"level": infer_level(title), "rating": rating, "count": count, "id": product_id},
        )


def _ensure_postgres_columns(sync_conn) -> None:
    """Opportunistic runtime migration for PostgreSQL: add any columns that
    were introduced after a table was first created. ADD COLUMN IF NOT EXISTS
    is a no-op on already-current schemas, so this is safe on every boot."""
    add_column = "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl}"
    for table, column, ddl in [
        ("users", "first_name", "VARCHAR(255)"),
        ("users", "last_name", "VARCHAR(255)"),
        ("user_cognitive_models", "recent_searches", "JSON"),
        ("user_cognitive_models", "recent_categories", "JSON"),
        ("recommendations", "behavior_explanation", "JSON"),
        ("products", "level", "VARCHAR(20)"),
        ("products", "rating", "FLOAT"),
        ("products", "rating_count", "INTEGER"),
    ]:
        sync_conn.exec_driver_sql(add_column.format(table=table, column=column, ddl=ddl))

    # Backfill rows created before the column existed (new rows get the
    # Python-side default, old ones load as NULL otherwise).
    sync_conn.exec_driver_sql(
        "UPDATE recommendations SET behavior_explanation = '[]'::json "
        "WHERE behavior_explanation IS NULL"
    )
    _backfill_product_meta(sync_conn)


def _ensure_sqlite_columns(sync_conn) -> None:
    """Opportunistic runtime migration for SQLite dev databases (SQLite has
    no ADD COLUMN IF NOT EXISTS, so introspect via PRAGMA instead)."""
    def _columns_of(sync_conn, table: str) -> set[str]:
        try:
            rows = sync_conn.exec_driver_sql(f"PRAGMA table_info('{table}')").fetchall()
            return {r[1] for r in rows}
        except Exception:
            return set()

    def _ensure_columns(sync_conn, table: str, cols_to_add: list[tuple[str, str]]) -> None:
        cols = _columns_of(sync_conn, table)
        for col, ddl in cols_to_add:
            if col not in cols:
                sync_conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    user_cols = _columns_of(sync_conn, "users")
    if "first_name" not in user_cols:
        sync_conn.exec_driver_sql("ALTER TABLE users ADD COLUMN first_name VARCHAR")
    if "last_name" not in user_cols:
        sync_conn.exec_driver_sql("ALTER TABLE users ADD COLUMN last_name VARCHAR")

    _ensure_columns(sync_conn, "user_cognitive_models", [
        ("recent_searches", "recent_searches JSON"),
        ("recent_categories", "recent_categories JSON"),
    ])
    _ensure_columns(sync_conn, "recommendations", [
        ("behavior_explanation", "behavior_explanation JSON"),
    ])
    # Backfill rows created before the column existed (new rows get the
    # Python-side default, old ones load as NULL otherwise).
    sync_conn.exec_driver_sql(
        "UPDATE recommendations SET behavior_explanation = '[]' WHERE behavior_explanation IS NULL"
    )

    # Course level + seeded rating/rating-count for products that predate
    # the columns.
    _ensure_columns(sync_conn, "products", [
        ("level", "level VARCHAR"),
        ("rating", "rating FLOAT"),
        ("rating_count", "rating_count INTEGER"),
    ])
    _backfill_product_meta(sync_conn)


async def init_db() -> None:
    """Create tables if they don't exist, then apply any missing-column
    migrations. Dialect-aware (SQLite dev DBs vs PostgreSQL production) and
    idempotent — safe on every startup. Swap for Alembic migrations if the
    project outlives the demo."""
    from sqlalchemy import text

    from app.db.session import engine

    dialect = engine.dialect.name

    async with engine.begin() as conn:
        if dialect == "postgresql":
            # pgvector must exist before create_all can create the
            # course_embeddings VECTOR column.
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create any missing tables
        await conn.run_sync(Base.metadata.create_all)

        if dialect == "postgresql":
            await conn.run_sync(_ensure_postgres_columns)
        else:
            await conn.run_sync(_ensure_sqlite_columns)
