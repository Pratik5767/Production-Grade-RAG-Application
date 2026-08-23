from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict):
    """
    AgentState defines the shared "memory" / state schema that flows through every node in a LangGraph graph. Each node receives this state, can read from it, and returns a (partial) update to it.
    """

    messages: Annotated[List[dict], operator.add]    # conversation history / maintaining the state of messages (HumanMessages, SystemMessages, etc.)
    current_query: str                              # the user's active question
    documents: List[str]                            # retrieved context chunks (RAG results)
    plan: List[str]                                 # list of steps the agent intends to execute
    status: str                                     # current phase of the agent (e.g. "retrieving", "done")
    final_answer: str                               # the final response to return to the user
