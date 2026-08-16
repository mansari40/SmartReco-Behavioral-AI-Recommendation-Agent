import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SyncStatus(str, enum.Enum):
    """Tracks whether the SQL row and the vector-DB entry are in sync.
    Dual-write is graded explicitly — this makes drift visible instead of silent."""
    SYNCED = "synced"
    PENDING = "pending"
    FAILED = "failed"


class Product(Base):
    __tablename__ = "products"
    # Title is the catalog's natural key: seed_catalog() and the admin API
    # both key on it, and the migration enforces it on existing DBs with the
    # same-named index (scripts/dedupe_catalog.py).
    __table_args__ = (Index("uq_products_title", "title", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)

    # Store metadata: course level + seeded rating (see services/catalog_meta).
    # NULL only for rows created before these columns existed; init_db()
    # backfills them deterministically.
    level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating_count: Mapped[int | None] = mapped_column(nullable=True)

    # Mirrors the ID used in the vector DB (Chroma) so the two stores can be
    # cross-referenced and reconciled.
    vector_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    sync_status: Mapped[SyncStatus] = mapped_column(
        Enum(SyncStatus), default=SyncStatus.PENDING, nullable=False
    )
    sync_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_embedding_text(self) -> str:
        """Canonical text representation used for embeddings — keep this
        stable so re-embeds on edit are meaningful diffs, not noise."""
        return f"{self.title}. Category: {self.category}. {self.description}"