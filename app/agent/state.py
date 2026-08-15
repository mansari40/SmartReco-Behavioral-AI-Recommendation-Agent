"""
Shared state that flows through every node in the recommendation agent
graph. Each node reads what it needs and returns a partial update — LangGraph
merges these into the running state automatically.
"""
from typing import TypedDict


class AgentState(TypedDict, total=False):
    # input
    user_id: str
    trigger_reason: str

    # populated by model_user node 
    cognitive_model: dict  # serialized UserCognitiveModel fields

    # populated by retrieve node
    retrieved_candidates: list[dict]  # raw Chroma query results, normalized
    query_embedding: list[float]      # computed by retrieve, reused by a relax retry
    retrieval_filter_applied: bool    # whether a category metadata filter was used

    # populated by assess/retry nodes (deterministic adaptive retrieval)
    retrieval_quality: str  # "good" | "low" — deterministic quality decision
    retrieval_adjusted: bool  # True if the filter-relaxing retry ran

    # populated by evaluate/filter nodes
    evaluated_candidates: list[dict]  # candidates scored against the cognitive model
    filtered_candidates: list[dict]   # top-N kept after evaluation

    # populated by generate node
    narrative: str
    recommended_products: list[dict]  
    persuasion_strategy: str
    confidence: float
    reasoning_chain: list[str]
    alternatives_considered: list[str]
    behavior_explanation: list[str]  # user-safe explanation, built from observable behavior

    # populated by reflect node
    should_regenerate: bool
    reflection_feedback: str
    regenerate_count: int  # guards against infinite reflect