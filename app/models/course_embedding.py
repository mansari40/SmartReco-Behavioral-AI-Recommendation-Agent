"""
Course embeddings for semantic retrieval, persisted in PostgreSQL via pgvector.

This table is the production vector store (one row per course). The column
types are chosen at import time from the configured DATABASE_URL:
PostgreSQL uses a real VECTOR column (with the pgvector extension) plus JSONB
metadata; SQLite (local dev / tests, where the Chroma fallback store is used)
gets a plain JSON column so the table definition stays harmless there.
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.db.base import Base

if settings.database_url.startswith("postgres"):
    from sqlalchemy.dialects.postgresql import JSONB

    from pgvector.sqlalchemy import Vector

    EMBEDDING_TYPE = Vector(None)  # no fixed dimension: model/mock dims differ
    METADATA_TYPE = JSONB
else:
    from sqlalchemy import JSON

    EMBEDDING_TYPE = JSON
    METADATA_TYPE = JSON


class CourseEmbedding(Base):
    __tablename__ = "course_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Stable cross-reference key shared with Product.vector_id so the SQL and
    # vector stores can be reconciled.
    vector_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)

    # "metadata" is reserved in SQLAlchemy 2.0 declarative, so the attribute
    # is metadata_ mapped to a column actually named "metadata".
    metadata_: Mapped[dict] = mapped_column("metadata", METADATA_TYPE, nullable=False, default=dict)

    embedding: Mapped[list] = mapped_column(EMBEDDING_TYPE, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
