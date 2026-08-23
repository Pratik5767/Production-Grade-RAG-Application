from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # local memmory saver and use simple conversational memmory
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


# Initialising the State Graph
workflow = StateGraph(AgentState)


# Defines the nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)


# Define the edges and routing
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """

    if state['current_query'] == 'CONVERSATIONAL':
        return "responder"
    return "retriever"


# setting starting point
workflow.set_entry_point("planner")


# Conditional Edge: Planner -> Router -> (Retriever OR Responder)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)


workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)


# --- MEMORY UPGRADE ---
# MemorySaver allows the agent to remember conversations based on 'thread_id'
checkpointer = MemorySaver()


# Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)
