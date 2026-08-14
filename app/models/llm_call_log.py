import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class LLMCallLog(Base):
    """
    A record of every real LLM/embedding call the app makes — this is what
    the Agent Console reads from. Every number shown there traces back to
    an actual row here; nothing on that page is invented.
    """
    __tablename__ = "llm_call_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    call_type: Mapped[str] = mapped_column(String(20))  # "chat" | "embedding"
    model: Mapped[str] = mapped_column(String(120))
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)  # honest flag for dev-mode mock calls

    latency_ms: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )