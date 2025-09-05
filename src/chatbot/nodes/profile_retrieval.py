"""
Profile retrieval node using LangGraph Store.
Retrieves user conversation profile from LangGraph's native memory store.
"""

from typing import List
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from src.chatbot.state import ConversationState


def search_user_profile(state: ConversationState, *, store) -> ConversationState:
    """Retrieve user profile from LangGraph Store."""
    try:
        messages = state["messages"]
        if not messages:
            return {"retrieved_conversations": []}

        # Get the latest human message as search query
        human_messages = [
            msg for msg in messages if hasattr(msg, "type") and msg.type == "human"
        ]
        if not human_messages:
            return {"retrieved_conversations": []}

        search_query = str(human_messages[-1].content).strip()
        print(f"🔍 Searching profile: '{search_query[:50]}...'")

        # Search user profile in LangGraph Store
        namespace = ("user", "demo_user")  # Simple demo namespace
        results = store.search(namespace, query=search_query, limit=3)

        if results:
            # Extract profile info as conversation context - prioritize facts
            profile_items = []
            for item in results:
                # Put facts first (these have the specific details)
                if "facts" in item.value and item.value["facts"]:
                    profile_items.extend(item.value["facts"])

                # Add interests/topics with more detail
                if "topics" in item.value and item.value["topics"]:
                    profile_items.append(
                        f"User is interested in: {', '.join(item.value['topics'])}"
                    )

            print(f"✅ Found profile info: {len(profile_items)} items")
            return {
                "retrieved_conversations": profile_items[:4]
            }  # More items, facts first
        else:
            print("ℹ️ No profile found")
            return {"retrieved_conversations": []}

    except Exception as e:
        print(f"❌ Profile retrieval error: {e}")
        return {"retrieved_conversations": []}


if __name__ == "__main__":
    """Demo the profile retrieval node."""
    from dotenv import load_dotenv

    load_dotenv()

    print("🧪 Testing Profile Retrieval Node...")

    # Create demo store with sample profile
    store = InMemoryStore()

    # Add sample profile data
    namespace = ("user", "demo_user")
    store.put(
        namespace,
        "interests",
        {
            "topics": ["Python", "AI", "Docker"],
            "preferences": "prefers concise explanations",
            "facts": ["Uses Docker for containerization", "Learning about LangChain"],
        },
    )

    # Test retrieval
    test_state = {"messages": [HumanMessage(content="Tell me about Python")]}
    result = search_user_profile(test_state, store=store)

    print(f"✅ Profile items found: {len(result['retrieved_conversations'])}")
    for item in result["retrieved_conversations"]:
        print(f"  - {item}")
