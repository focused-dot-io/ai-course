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


if __name__ == "__main__":
    """Demo the routing node."""
    from langchain_core.messages import HumanMessage

    print("🧪 Testing Routing Node...")

    # Test normal message routing
    normal_state = {
        "messages": [HumanMessage(content="Tell me about Python")],
        "retrieved_conversations": [],
    }

    route1 = should_store_conversation(normal_state)
    print(f"✅ Normal message routes to: {route1}")

    # Test quit message routing
    quit_state = {
        "messages": [HumanMessage(content="quit")],
        "retrieved_conversations": [],
    }

    route2 = should_store_conversation(quit_state)
    print(f"✅ Quit message routes to: {route2}")

    # Test exit message routing
    exit_state = {
        "messages": [HumanMessage(content="exit")],
        "retrieved_conversations": [],
    }

    route3 = should_store_conversation(exit_state)
    print(f"✅ Exit message routes to: {route3}")

    # Test edge case - empty messages
    empty_state = {"messages": [], "retrieved_conversations": []}

    route4 = should_store_conversation(empty_state)
    print(f"✅ Empty messages routes to: {route4}")
