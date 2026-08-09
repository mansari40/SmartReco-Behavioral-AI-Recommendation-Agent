import enum
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class DecisionStage(str, enum.Enum):
    AWARENESS = "awareness"
    INTEREST = "interest"
    EVALUATION = "evaluation"
    DECISION = "decision"
    POST_PURCHASE = "post_purchase"


class PriceSensitivity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UserCognitiveModel(Base):
    """
    A persistent, evolving structured representation of what the agent
    believes about a user — updated by an LLM call (via Mesh) after each
    meaningful batch of events, not recomputed from scratch every time.
    This is what the `model_user` LangGraph node reads and writes.
    """
    __tablename__ = "user_cognitive_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), unique=True, index=True, nullable=False
    )

    stated_intents: Mapped[list] = mapped_column(JSON, default=list)
    inferred_intents: Mapped[list] = mapped_column(JSON, default=list)

    decision_stage: Mapped[DecisionStage] = mapped_column(
        Enum(DecisionStage), default=DecisionStage.AWARENESS
    )
    purchase_readiness: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 - 1.0
    price_sensitivity: Mapped[PriceSensitivity] = mapped_column(
        Enum(PriceSensitivity), default=PriceSensitivity.MEDIUM
    )

    detected_objections: Mapped[list] = mapped_column(JSON, default=list)
    brand_affinity: Mapped[list] = mapped_column(JSON, default=list)
    category_affinity: Mapped[list] = mapped_column(JSON, default=list)

    session_arc: Mapped[str] = mapped_column(Text, default="")

    # Bookkeeping used by trigger logic — avoids firing an LLM call on
    # every single event by tracking how many events have arrived since
    # the model was last actually updated.
    last_event_count_at_update: Mapped[int] = mapped_column(default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    user: Mapped["User"] = relationship(back_populates="cognitive_model")