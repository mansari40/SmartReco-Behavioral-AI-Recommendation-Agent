import enum
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class EventType(str, enum.Enum):
    PAGE_VIEW = "page_view"
    PRODUCT_VIEW = "product_view"
    SEARCH = "search"
    CLICK = "click"
    TIME_SPENT = "time_spent"
    ADD_TO_CART = "add_to_cart"
    CHECKOUT_START = "checkout_start"


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), nullable=False)
    product_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("products.id"), nullable=True)

    # Free-form payload: search query text, time_spent seconds, click target, etc.
    event_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    user: Mapped["User"] = relationship(back_populates="events")