"""
LangGraph conversation graph definition.

This module defines the conversation flow for the chatbot using LangGraph Studio compatible structure.
"""

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from src.chatbot.nodes import generate_response


def create_conversation_graph():
    """
    Create and compile the conversation graph with checkpointing.

    Returns:
        Compiled LangGraph StateGraph with memory for conversation persistence
    """
    # Create the workflow
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("respond", generate_response)

    # Define the flow
    workflow.add_edge(START, "respond")
    workflow.add_edge("respond", END)

    # Compile with checkpointer for conversation memory
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
