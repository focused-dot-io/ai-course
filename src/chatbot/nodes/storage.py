"""
Storage node for LangGraph that handles conversation summarization and storage.
Saves conversation summaries when conversations end.
"""

from typing import Dict, List, Optional
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from src.chatbot.simple_vector_store import SimpleVectorStore
from src.chatbot.state import ConversationState

load_dotenv()


SUMMARIZATION_PROMPT = """Create a structured summary of this conversation as a list of things discussed, with topics at the end for better searchability.

Conversation:
{conversation_text}

Summary:
Here are the things we discussed:
- [list the main points]
- [list the main points]
- [list the main points]

Topics: [list main topics separated by commas]"""


def summarize_conversation(messages: List[BaseMessage]) -> Optional[str]:
    """Simple conversation summarizer - returns combined summary with topics."""
    if len(messages) < 2:
        return None

    # Convert to text
    conversation_text = ""
    for msg in messages:
        if hasattr(msg, "type"):
            role = "Human" if msg.type == "human" else "AI"
            conversation_text += f"{role}: {msg.content}\n"

    try:
        llm = init_chat_model("openai:gpt-4o-mini", temperature=0.3)
        prompt = SUMMARIZATION_PROMPT.format(conversation_text=conversation_text)
        response = llm.invoke(prompt)
        summary = response.content.strip()

        return summary if summary else None

    except Exception as e:
        print(f"❌ Summarization error: {e}")
        return None


def store_conversation_summary(state: ConversationState) -> ConversationState:
    """Simple storage: summarize and store using LangChain pgvector."""
    try:
        messages = state["messages"]

        # Filter out system messages
        filtered_messages = [
            msg
            for msg in messages
            if not (hasattr(msg, "type") and msg.type == "system")
        ]

        if len(filtered_messages) >= 2:
            print("📝 Storing conversation...")

            # Create simple summary
            summary = summarize_conversation(filtered_messages)

            if summary:
                # Store using LangChain pgvector
                vector_store = SimpleVectorStore()
                vector_store.store_conversation(summary=summary)
                msg = "✅ Conversation saved!"
            else:
                msg = "ℹ️ Conversation too short to save."
        else:
            msg = "ℹ️ Conversation too short to save."

        # Add confirmation message
        return {"messages": [SystemMessage(content=msg)]}

    except Exception as e:
        print(f"❌ Storage error: {e}")
        return {"messages": [SystemMessage(content="❌ Could not save conversation.")]}


if __name__ == "__main__":
    """Demo the storage node."""
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, AIMessage

    load_dotenv()

    print("🧪 Testing Storage Node...")

    # Test conversation summarization
    test_messages = [
        HumanMessage(content="What's the difference between Docker and VMs?"),
        AIMessage(
            content="Docker uses containerization which is more lightweight than VMs. Containers share the host OS kernel while VMs have their own OS."
        ),
        HumanMessage(content="Which is better for microservices?"),
        AIMessage(
            content="Docker is generally preferred for microservices because it's lighter, faster to start, and easier to scale."
        ),
    ]

    # Test summarization function directly
    summary = summarize_conversation(test_messages)
    if summary:
        print(f"✅ Generated summary:\n{summary[:200]}...")
    else:
        print("❌ No summary generated")

    # Test full storage node
    test_state = {"messages": test_messages, "retrieved_conversations": []}

    result = store_conversation_summary(test_state)
    print(f"✅ Storage result: {result['messages'][-1].content}")
