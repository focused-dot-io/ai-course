"""
Response generation node for the conversation chatbot.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from typing import List

from src.chatbot.state import ConversationState


SYSTEM_TEMPLATE = """You are a helpful and friendly AI assistant. You can:
- Answer questions on any topic
- Maintain context from our conversation
- Provide detailed explanations when asked
- Be creative and engaging in your responses

Keep your responses conversational and helpful.

Relevant information (there may not be any):
{conversations}"""


def generate_response(state: ConversationState) -> ConversationState:
    """
    Generate AI response based on conversation history.

    Args:
        state: Current conversation state with message history

    Returns:
        Updated state with AI response added
    """
    # Initialize the LLM with streaming enabled
    llm = init_chat_model("openai:gpt-4o", temperature=0.7, streaming=True)

    # Create the system prompt
    system_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            (
                "placeholder",
                "{messages}",
            ),  # This expands to the full conversation history
        ]
    )

    # Prepare conversations list
    retrieved_conversations = state.get("retrieved_conversations", [])
    conversations = "\n".join(f"- {conv}" for conv in retrieved_conversations)

    # Create the chain
    chain = system_prompt | llm

    # Generate response
    response = chain.invoke(
        {"messages": state["messages"], "conversations": conversations}
    )

    # Return updated state
    return {"messages": [response]}


if __name__ == "__main__":
    """Demo the response generation node."""
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, AIMessage

    load_dotenv()

    print("🧪 Testing Response Generation Node...")

    # Test with empty retrieved conversations
    test_state = {
        "messages": [HumanMessage(content="What is machine learning?")],
        "retrieved_conversations": [],
    }

    result = generate_response(test_state)
    print(f"✅ Generated response: {result['messages'][0].content[:100]}...")

    # Test with retrieved conversations
    print("\n🔍 Testing with context...")
    test_state_with_context = {
        "messages": [HumanMessage(content="Tell me more about AI")],
        "retrieved_conversations": [
            "User asked about machine learning basics and supervised vs unsupervised learning",
            "Discussion about neural networks and deep learning applications",
        ],
    }

    result2 = generate_response(test_state_with_context)
    print(f"✅ Response with context: {result2['messages'][0].content[:100]}...")
