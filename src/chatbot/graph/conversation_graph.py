"""
LangGraph conversation graph definition.

This module defines the conversation flow for the chatbot using LangGraph Studio compatible structure.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.chatbot.state import ConversationState
from src.chatbot.nodes import generate_response
from src.chatbot.nodes.retrieval import search_relevant_conversations
from src.chatbot.nodes.storage import store_conversation_summary
from src.chatbot.nodes.routing import should_store_conversation


def create_conversation_graph():
    """
    Create and compile the conversation graph with memory, retrieval, and storage.

    Returns:
        Compiled LangGraph with checkpointer for conversation memory, retrieval, and storage
    """
    # Create the workflow
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("search_conversations", search_relevant_conversations)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("store_conversation", store_conversation_summary)

    # Flow: check for quit first, then search if needed
    workflow.add_conditional_edges(
        START,
        should_store_conversation,
        {
            "store_conversation": "store_conversation",
            "generate_response": "search_conversations",
        },
    )

    # After search, generate response
    workflow.add_edge("search_conversations", "generate_response")

    # Both storage and response end the conversation
    workflow.add_edge("store_conversation", END)
    workflow.add_edge("generate_response", END)

    # Compile with checkpointer for conversation memory
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
