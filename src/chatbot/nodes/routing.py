"""
Routing node for LangGraph that decides the next step in conversation flow.
Determines whether to store conversation or generate response.
"""

from typing import List
from src.chatbot.state import ConversationState


def should_store_conversation(state: ConversationState) -> str:
    """Check if user said quit, otherwise generate response."""
    messages = state["messages"]

    # Get last message content
    if messages and hasattr(messages[-1], "content"):
        last_message = messages[-1].content.lower().strip()
        if last_message in ["quit", "exit"]:
            return "store_conversation"

    return "generate_response"
