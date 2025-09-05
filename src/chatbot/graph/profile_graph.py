"""
LangGraph conversation graph using native Store for memory.
Demonstrates profile-based memory management.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.chatbot.state import ConversationState
from src.chatbot.nodes import generate_response
from src.chatbot.nodes.profile_retrieval import search_user_profile
from src.chatbot.nodes.profile_storage import update_user_profile
from src.chatbot.nodes.routing import should_store_conversation


def create_profile_graph():
    """
    Create conversation graph with LangGraph Store memory.

    Returns:
        Compiled LangGraph with checkpointer and native memory store
    """
    # Create the workflow
    workflow = StateGraph(ConversationState)

    # Add nodes (reusing routing and response from original)
    workflow.add_node("search_profile", search_user_profile)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("update_profile", update_user_profile)

    # Same flow as vector approach
    workflow.add_conditional_edges(
        START,
        should_store_conversation,
        {
            "store_conversation": "update_profile",
            "generate_response": "search_profile",
        },
    )

    # Always update profile after generating response
    workflow.add_edge("search_profile", "generate_response")
    workflow.add_edge("generate_response", "update_profile")
    workflow.add_edge("update_profile", END)

    # Compile with both checkpointer AND store
    checkpointer = MemorySaver()
    store = InMemoryStore()  # Use DB-backed store in production

    return workflow.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    """Demo the profile-based graph."""
    from langchain_core.messages import HumanMessage
    import uuid

    print("🧪 Testing Profile Graph...")

    graph = create_profile_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Test conversation
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="I love working with Python and APIs")]},
        config=config,
    )

    print(f"✅ Response generated")
    print(f"Profile items: {len(result1.get('retrieved_conversations', []))}")

    # Test profile storage
    result2 = graph.invoke({"messages": [HumanMessage(content="quit")]}, config=config)

    print(f"✅ Profile storage: {result2['messages'][-1].content}")
