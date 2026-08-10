from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.event import EventType


class EventIn(BaseModel):
    """A single tracked event, as sent by the batched frontend tracker."""
    event_type: EventType
    product_id: str | None = None
    event_metadata: dict = {}
    client_timestamp: datetime | None = None


class EventBatchIn(BaseModel):
    """The frontend tracker flushes a batch, not one event per request —
    this is the payload shape for POST /api/events."""
    events: list[EventIn]


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    event_type: EventType
    product_id: str | None
    event_metadata: dict
    created_at: datetime