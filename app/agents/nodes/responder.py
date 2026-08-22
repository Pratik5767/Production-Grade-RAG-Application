from app.agents.state import AgentState
from app.config import settings
from langchain_groq import ChatGroq
import logfire


# Direct Groq call - the LLM Gateway (Portkey routing/fallback/caching) arrives in
llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    """

    # getting the current query (conversational or technical) 
    query = state["current_query"]

    # Get the conversation history
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    # fetching the current user message
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    # Simple conversational query (no retrieval)
    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")

        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY: {history}

        LATEST MESSAGE: "{user_msg}"
        """
    else:
        # Technical/RAG query - (retrieval needed)
        logfire.info("Generating technical RAG response.")

        # Cap context size to prevent LLM token/rate limits
        max_context_chars = 25000
        full_context = ""

        # Pack in as many retrieved docs as fit within the char budget
        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT: {full_context}

        CONVERSATION HISTORY: {history}

        USER QUESTION: "{user_msg}"
        """

    # Wrap the actual LLM call in a tracing span so we can see latency/failures in Logfire
    with logfire.span("✍️ LLM Synthesis"):
        try:
            content = llm.invoke(prompt).content
            logfire.info("✅ Response synthesis via LLM")

            return {
                "final_answer": content,
                "status": "Response Generated",
                "plan": state["plan"],
                "messages": [{"role": "assistant", "content": content}]
            }
        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
