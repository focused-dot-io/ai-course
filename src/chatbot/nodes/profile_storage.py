"""
Profile storage node using LangGraph Store.
Updates user conversation profile in LangGraph's native memory store.
"""

from typing import List, Optional
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

from src.chatbot.state import ConversationState

load_dotenv()

PROFILE_UPDATE_PROMPT = """Update the user profile based on this conversation.

Current Profile:
{current_profile}

Recent Conversation:
{conversation_text}

Return a JSON profile with these fields:
- topics: list of user interests/topics discussed
- preferences: user communication preferences  
- facts: list of specific facts about the user (names, details, personal info)

Focus on concrete facts and details. Include specific names, relationships, and personal details that help remember who this person is."""


def extract_profile_info(
    messages: List[BaseMessage], current_profile: dict
) -> Optional[dict]:
    """Extract profile information from conversation."""
    if len(messages) < 2:
        return None

    # Convert conversation to text
    conversation_text = ""
    for msg in messages:
        if hasattr(msg, "type") and msg.type != "system":
            role = "Human" if msg.type == "human" else "AI"
            conversation_text += f"{role}: {msg.content}\n"

    try:
        llm = init_chat_model("openai:gpt-4o-mini", temperature=0.1)
        prompt = PROFILE_UPDATE_PROMPT.format(
            current_profile=current_profile, conversation_text=conversation_text
        )
        response = llm.invoke(prompt)

        # Simple profile parsing (could use structured output in real app)
        content = response.content.strip()
        if content and "{" in content:
            import json

            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            profile_json = content[start:end]
            return json.loads(profile_json)

        return None
    except Exception as e:
        print(f"❌ Profile extraction error: {e}")
        return None


def update_user_profile(state: ConversationState, *, store) -> ConversationState:
    """Update user profile in LangGraph Store."""
    try:
        messages = state["messages"]

        # Filter out system messages
        filtered_messages = [
            msg
            for msg in messages
            if not (hasattr(msg, "type") and msg.type == "system")
        ]

        if len(filtered_messages) >= 2:
            print("📝 Updating user profile...")

            # Get current profile
            namespace = ("user", "demo_user")
            current_items = store.search(namespace)
            current_profile = current_items[0].value if current_items else {}

            # Extract new profile info
            updated_profile = extract_profile_info(filtered_messages, current_profile)

            if updated_profile:
                # Update profile in store
                store.put(namespace, "profile", updated_profile)
                msg = "✅ Profile updated!"
            else:
                msg = "ℹ️ No profile updates needed."
        else:
            msg = "ℹ️ Conversation too short to update profile."

        return {"messages": [SystemMessage(content=msg)]}

    except Exception as e:
        print(f"❌ Profile storage error: {e}")
        return {"messages": [SystemMessage(content="❌ Could not update profile.")]}


if __name__ == "__main__":
    """Demo the profile storage node."""
    from langchain_core.messages import HumanMessage, AIMessage

    print("🧪 Testing Profile Storage Node...")

    # Create demo store
    store = InMemoryStore()

    # Test profile update
    test_messages = [
        HumanMessage(content="I'm really interested in learning Docker"),
        AIMessage(
            content="Docker is great for containerization! What specific aspects interest you?"
        ),
        HumanMessage(content="I want to use it for Python web development"),
    ]

    test_state = {"messages": test_messages, "retrieved_conversations": []}
    result = update_user_profile(test_state, store=store)

    print(f"✅ Storage result: {result['messages'][-1].content}")

    # Check what was stored
    namespace = ("user", "demo_user")
    stored_items = store.search(namespace)
    if stored_items:
        print(f"📋 Stored profile: {stored_items[0].value}")
