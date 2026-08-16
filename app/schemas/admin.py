"""
Response models for the read-only admin API (Overview / Users / Observability).
Every field maps to an actual row or an aggregation of actual rows in the
database — the admin pages never invent numbers, they just read the truth.
"""
from datetime import datetime

from pydantic import BaseModel


class RecentActivityItem(BaseModel):
    id: str
    user_email: str
    event_type: str
    product_title: str | None
    summary: str
    created_at: datetime


class VectorSyncHealth(BaseModel):
    total: int
    synced: int
    pending: int
    failed: int
    sample_errors: list[dict] = []


class CostEstimate(BaseModel):
    """Estimated spend from real token counts and documented blended rates."""
    total_tokens: int
    estimated_usd: float
    basis: str


class AgentRunSummary(BaseModel):
    runs_24h: int
    runs_with_recommendation: int
    avg_pipeline_seconds: float
    llm_calls_24h: int
    total_tokens_24h: int
    estimated_cost_usd_24h: float


class NameCount(BaseModel):
    name: str
    count: int


class OverviewData(BaseModel):
    generated_at: datetime
    window_hours: int
    total_users: int
    active_users_24h: int
    model_calls_24h: int
    real_calls_24h: int
    mock_calls_24h: int
    success_rate_24h: float
    avg_latency_ms_24h: float
    recommendations_24h: int
    agent_runs: AgentRunSummary
    cost: CostEstimate
    vector_sync: VectorSyncHealth
    recent_activity: list[RecentActivityItem]
    decision_stage_distribution: dict[str, int]
    price_sensitivity_distribution: dict[str, int]
    avg_purchase_readiness: float
    cognitive_models_count: int
    top_inferred_intents: list[NameCount]
    top_category_affinity: list[NameCount]


class UserAdminListItem(BaseModel):
    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None
    role: str
    created_at: datetime
    events_count: int
    recommendations_count: int
    last_event_at: datetime | None = None
    feedback_up: int
    feedback_down: int
    has_active_recommendation: bool
    active_trigger_reason: str | None = None
    active_confidence: float | None = None
    decision_stage: str | None = None
    purchase_readiness: float | None = None
    price_sensitivity: str | None = None
    top_inferred_intents: list[str] = []


class UsersPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[UserAdminListItem]


class RecommendationBrief(BaseModel):
    id: str
    created_at: datetime
    is_active: bool
    feedback: str | None
    trigger_reason: str
    confidence: float
    persuasion_strategy: str
    product_count: int
    narrative: str


class EventBrief(BaseModel):
    id: str
    event_type: str
    product_title: str | None
    summary: str
    created_at: datetime


class CognitiveProfile(BaseModel):
    stated_intents: list[str]
    inferred_intents: list[str]
    decision_stage: str
    purchase_readiness: float
    price_sensitivity: str
    detected_objections: list[str]
    brand_affinity: list[str]
    category_affinity: list[str]
    recent_searches: list[str]
    recent_categories: list[str]
    session_arc: str
    updated_at: datetime | None


class UserDetailData(BaseModel):
    id: str
    email: str
    first_name: str | None
    last_name: str | None
    role: str
    created_at: datetime
    events_count: int
    recommendations_count: int
    feedback_up: int
    feedback_down: int
    cognitive_profile: CognitiveProfile | None = None
    recommendations: list[RecommendationBrief] = []
    recent_events: list[EventBrief] = []


class AgentRunRecommendation(BaseModel):
    id: str
    user_email: str
    trigger_reason: str
    persuasion_strategy: str
    confidence: float
    product_count: int
    created_at: datetime


class AgentRunDetail(BaseModel):
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    llm_calls: int
    chat_calls: int
    embedding_calls: int
    models: list[str]
    total_tokens: int
    estimated_cost_usd: float
    success: bool
    error: str | None = None
    recommendation: AgentRunRecommendation | None = None


class ObservabilityCall(BaseModel):
    id: str
    call_type: str
    model: str
    is_mock: bool
    latency_ms: int
    total_tokens: int | None
    success: bool
    error: str | None
    created_at: datetime
    estimated_cost_usd: float


class ObservabilityData(BaseModel):
    generated_at: datetime
    window_hours: int
    calls_24h: int
    real_calls_24h: int
    mock_calls_24h: int
    success_rate_24h: float
    avg_latency_ms_24h: float
    total_tokens_24h: int
    estimated_cost_usd_24h: float
    model_breakdown: dict[str, int]
    runs: list[AgentRunDetail]
    recent_calls: list[ObservabilityCall]