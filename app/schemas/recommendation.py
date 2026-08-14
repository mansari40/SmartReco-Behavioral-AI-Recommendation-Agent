from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    narrative: str
    products: list[dict]
    persuasion_strategy: str
    confidence: float
    reasoning_chain: list
    alternatives_considered: list
    behavior_explanation: list = Field(default_factory=list)
    trigger_reason: str
    feedback: str | None
    created_at: datetime


class FeedbackIn(BaseModel):
    feedback: str  # "up" or "down"