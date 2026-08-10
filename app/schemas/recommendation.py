from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    narrative: str
    products: list[dict]
    persuasion_strategy: str
    confidence: float
    reasoning_chain: list
    alternatives_considered: list
    trigger_reason: str
    created_at: datetime