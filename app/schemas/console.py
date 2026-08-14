from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CallLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    call_type: str
    model: str
    is_mock: bool
    latency_ms: int
    total_tokens: int | None
    success: bool
    error: str | None
    created_at: datetime


class ConsoleStats(BaseModel):
    calls_today: int
    real_calls_today: int
    mock_calls_today: int
    avg_latency_ms: float
    success_rate: float
    model_breakdown: dict[str, int]
    recent_calls: list[CallLogOut]