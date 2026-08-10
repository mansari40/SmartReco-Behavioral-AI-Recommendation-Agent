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

    # populated by reflect node
    should_regenerate: bool
    reflection_feedback: str
    regenerate_count: int  # guards against infinite reflect