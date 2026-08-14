import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class Recommendation(Base):
    """
    A stored agent output. `products` and `reasoning_chain` are JSON so the
    frontend "why am I seeing this" panel can render the agent's reasoning
    without re-deriving it. `is_active` keeps history while only surfacing
    the latest per user. `feedback` lets a user thumbs up/down a result —
    cheap to add, shows the system can learn from signal, not just generate.
    """
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)

    narrative: Mapped[str] = mapped_column(Text, nullable=False)

    # [{"product_id": "...", "reason": "..."}, ...]
    products: Mapped[list] = mapped_column(JSON, default=list)

    persuasion_strategy: Mapped[str] = mapped_column(String(120), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Step-by-step reasoning for the agent console (audit trail — never
    # rendered to users), and what the reflection node considered but
    # rejected — purely for auditability.
    reasoning_chain: Mapped[list] = mapped_column(JSON, default=list)
    alternatives_considered: Mapped[list] = mapped_column(JSON, default=list)

    # User-safe "why am I seeing this" — deterministic facts derived from
    # observable behavior (searches, categories viewed), never model
    # chain-of-thought. Rendered by the recommendation UI.
    behavior_explanation: Mapped[list] = mapped_column(JSON, default=list)

    trigger_reason: Mapped[str] = mapped_column(String(120), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    feedback: Mapped[str | None] = mapped_column(String(20), nullable=True)  # "up" | "down" | None

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    user: Mapped["User"] = relationship(back_populates="recommendations")