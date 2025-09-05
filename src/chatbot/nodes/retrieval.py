"""
Retrieval node for LangGraph that searches past conversations for context.
Enhances responses by finding relevant historical conversations.
"""

from typing import List
from langchain_core.messages import HumanMessage

from src.chatbot.simple_vector_store import SimpleVectorStore
from src.chatbot.state import ConversationState


def extract_search_query_from_messages(messages: List) -> str:
    """Just get the latest human message as the search query."""
    # Get the most recent human message
    human_messages = [
        msg for msg in messages if hasattr(msg, "type") and msg.type == "human"
    ]
    if not human_messages:
        return ""

    return str(human_messages[-1].content).strip()


def search_relevant_conversations(state: ConversationState) -> ConversationState:
    """Simple retrieval: search with the latest message and add context if found."""
    try:
        messages = state["messages"]
        search_query = extract_search_query_from_messages(messages)

        if not search_query:
            return state

        print(f"🔍 Searching: '{search_query[:50]}...'")

        # Search vector store using LangChain's pgvector
        vector_store = SimpleVectorStore()
        results = vector_store.search_conversations(
            query=search_query, limit=2, distance_threshold=0.8
        )

        if results:
            print(f"✅ Found context from {len(results)} past conversations")
            return {"retrieved_conversations": results}
        else:
            return {"retrieved_conversations": []}

    except Exception as e:
        print(f"❌ Error in retrieval: {e}")
        return {"retrieved_conversations": []}


if __name__ == "__main__":
    """Test the retrieval functionality."""
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage

    load_dotenv()

    print("🧪 Testing Simple Retrieval...")

    test_state = {"messages": [HumanMessage(content="How do I use Docker?")]}
    result_state = search_relevant_conversations(test_state)

    print(f"Original: {len(test_state['messages'])} messages")
    print(
        f"Retrieved conversations: {len(result_state.get('retrieved_conversations', []))}"
    )

    if result_state.get("retrieved_conversations"):
        print("✅ Context found!")
        for i, conv in enumerate(result_state["retrieved_conversations"], 1):
            print(f"  {i}. {conv[:50]}...")
    else:
        print("ℹ️ No context found")
