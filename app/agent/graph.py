"""
The recommendation agent, built as an explicit LangGraph workflow:

    model_user -> retrieve -> assess -> evaluate -> filter -> generate -> reflect -> store -> END
                                 │
                                 └─▶ retry (deterministic relax, only on poor quality)

Each node plays a distinct role in the pipeline. To keep LLM cost bounded,
only the calls that genuinely need a model make one:

    model_user  — LLM profile inference (1 chat call)
    retrieve    — query embedding (1 embedding call)
    assess      — deterministic retrieval-quality gate (no LLM)
    retry       — deterministic filter-relaxing re-query reusing the same
                  embedding (no LLM, zero extra AI calls)
    evaluate    — deterministic relevance scoring (no LLM)
    filter      — deterministic top-N selection (no LLM)
    generate    — LLM narrative generation (1 chat call — the single
                  generation per run)
    reflect     — deterministic quality/grounding gate (no LLM); never
                  requests regeneration
    store       — persistence (no LLM)

The conditional reflect/regenerate loop and MAX_REGENERATIONS cap are
retained for future re-enabling, but reflect deterministically returns
should_regenerate=False so a run always produces exactly one generation.

The genuinely adaptive part is the assess/retry branch: assess reads
deterministic signals (candidate count, best similarity) that are already
in state and either routes straight to evaluate (good quality, the normal
path) or to retry (poor quality with an over-constraining metadata filter).
retry re-searches without the filter using the exact embedding retrieve
already computed — an observable, bounded adjustment that adds no LLM calls.
"""
from langgraph.graph import END, StateGraph
from app.agent.nodes.model_user import model_user_node
from app.agent.nodes.retrieve import (
    MAX_RETRIEVAL_ATTEMPTS,
    retrieve_node,
    assess_retrieval_node,
    retry_retrieve_node,
)
from app.agent.state import AgentState
from app.agent.nodes.evaluate import evaluate_node
from app.agent.nodes.filter import filter_node
from app.agent.nodes.generate import generate_node
from app.agent.nodes.reflect import reflect_node
from app.agent.nodes.store import store_node

MAX_REGENERATIONS = 2


def _route_after_assess(state: AgentState) -> str:
    """Conditional edge: after the deterministic quality gate, either retry
    with the next progressive filter stage (newest category -> broader recent
    categories -> unfiltered) or proceed to evaluate. Bounded to
    MAX_RETRIEVAL_ATTEMPTS via retrieval_attempt — the final stage has no
    filter, so the last assess always routes to evaluate."""
    if state.get("retrieval_quality") == "good":
        return "evaluate"
    if state.get("retrieval_filter_applied") and state.get("retrieval_attempt", 1) < MAX_RETRIEVAL_ATTEMPTS:
        return "retry"
    return "evaluate"


def _should_loop_or_finish(state: AgentState) -> str:
    """Conditional edge: after reflect, either go back to generate (if the
    reflection said so, and we haven't hit the retry cap) or move on to store."""
    if state.get("should_regenerate") and state.get("regenerate_count", 0) < MAX_REGENERATIONS:
        return "generate"
    return "store"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("model_user", model_user_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("assess", assess_retrieval_node)
    graph.add_node("retry", retry_retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("filter", filter_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("store", store_node)

    graph.set_entry_point("model_user")
    graph.add_edge("model_user", "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges(
        "assess", _route_after_assess, {"evaluate": "evaluate", "retry": "retry"}
    )
    graph.add_edge("retry", "assess")
    graph.add_edge("evaluate", "filter")
    graph.add_edge("filter", "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges("reflect", _should_loop_or_finish, {"generate": "generate", "store": "store"})
    graph.add_edge("store", END)

    return graph.compile()


agent_graph = build_graph()