"""
Shared state definition for the conversation chatbot.
"""

from typing import List
from langgraph.graph import MessagesState


class ConversationState(MessagesState):
    """Extended state that includes retrieved conversations."""

    retrieved_conversations: List[str] = []
