"""
Declarative base + a single import point that pulls in every model so
Base.metadata is complete when init_db() runs create_all().
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import models so they register with Base.metadata. Kept at the bottom of
# the file to avoid circular imports between the models and Base itself.
from app.models import user, product, event, cognitive_model, recommendation, llm_call_log  # noqa: E402,F401


async def init_db() -> None:
    """Create tables if they don't exist. Fine for a hackathon timeline;
    swap for Alembic migrations if the project outlives the demo."""
    from app.db.session import engine

    async with engine.begin() as conn:
        # Create any missing tables
        await conn.run_sync(Base.metadata.create_all)

        # Ensure new nullable columns are present on existing tables. This
        # is a tiny, opportunistic runtime migration to avoid requiring
        # Alembic for this demo. Only add columns if they're missing.
        def _ensure_user_columns(sync_conn):
            try:
                cur = sync_conn.exec_driver_sql("PRAGMA table_info('users')")
                rows = cur.fetchall()
                cols = [r[1] for r in rows]
            except Exception:
                cols = []

            if 'first_name' not in cols:
                sync_conn.exec_driver_sql("ALTER TABLE users ADD COLUMN first_name VARCHAR")
            if 'last_name' not in cols:
                sync_conn.exec_driver_sql("ALTER TABLE users ADD COLUMN last_name VARCHAR")

            def _ensure_columns(sync_conn, table, cols_to_add):
                try:
                    cur = sync_conn.exec_driver_sql(f"PRAGMA table_info('{table}')")
                    rows = cur.fetchall()
                    cols = {r[1] for r in rows}
                except Exception:
                    cols = set()
                for col, ddl in cols_to_add:
                    if col not in cols:
                        sync_conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {ddl}")

            _ensure_columns(sync_conn, 'user_cognitive_models', [
                ('recent_searches', 'recent_searches JSON'),
                ('recent_categories', 'recent_categories JSON'),
            ])
            _ensure_columns(sync_conn, 'recommendations', [
                ('behavior_explanation', 'behavior_explanation JSON'),
            ])
            # Backfill rows created before the column existed (new rows get
            # the Python-side default, old ones load as NULL otherwise).
            sync_conn.exec_driver_sql(
                "UPDATE recommendations SET behavior_explanation = '[]' WHERE behavior_explanation IS NULL"
            )

            # Course level + seeded rating/rating-count for products that
            # predate the columns. Deterministic: level from title keywords,
            # rating from a stable hash of the product id.
            _ensure_columns(sync_conn, 'products', [
                ('level', 'level VARCHAR'),
                ('rating', 'rating FLOAT'),
                ('rating_count', 'rating_count INTEGER'),
            ])
            from app.services.catalog_meta import infer_level, seed_rating
            for row in sync_conn.exec_driver_sql(
                "SELECT id, title FROM products WHERE level IS NULL OR rating IS NULL"
            ).fetchall():
                product_id, title = row[0], row[1]
                rating, count = seed_rating(product_id)
                sync_conn.exec_driver_sql(
                    "UPDATE products SET level = ?, rating = ?, rating_count = ? WHERE id = ?",
                    (infer_level(title), rating, count, product_id),
                )

        await conn.run_sync(_ensure_user_columns)