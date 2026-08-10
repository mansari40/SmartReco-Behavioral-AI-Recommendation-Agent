"""
The recommendation agent, built as an explicit LangGraph workflow:

    model_user -> retrieve -> evaluate -> filter -> generate -> reflect
                                                          ^          |
                                                          +---(loop)-+
                                                                     |
                                                                  store -> END

Each node is a stub for now (returns fixed placeholder data) so the graph's
*shape* — the wiring, the conditional reflect/regenerate loop, the max-retry
guard — can be proven correct before any real LLM/retrieval logic goes in.
Real logic gets filled in node by node in the next steps.
"""
from langgraph.graph import END, StateGraph
from app.agent.nodes.model_user import model_user_node
from app.agent.state import AgentState
from app.agent.nodes.retrieve import retrieve_node
from app.agent.nodes.evaluate import evaluate_node
from app.agent.nodes.filter import filter_node
from app.agent.nodes.generate import generate_node
from app.agent.nodes.reflect import reflect_node
from app.agent.nodes.store import store_node

MAX_REGENERATIONS = 2




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
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("filter", filter_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("store", store_node)

    graph.set_entry_point("model_user")
    graph.add_edge("model_user", "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("evaluate", "filter")
    graph.add_edge("filter", "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges("reflect", _should_loop_or_finish, {"generate": "generate", "store": "store"})
    graph.add_edge("store", END)

    return graph.compile()


agent_graph = build_graph()