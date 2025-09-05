"""
Response generation node for the conversation chatbot.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState


def generate_response(state: MessagesState) -> MessagesState:
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
            (
                "system",
                """You are a helpful and friendly AI assistant. You can:
- Answer questions on any topic
- Maintain context from our conversation
- Provide detailed explanations when asked
- Be creative and engaging in your responses

Keep your responses conversational and helpful.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    # Create the chain
    chain = system_prompt | llm

    # Generate response
    response = chain.invoke({"messages": state["messages"]})

    # Return updated state
    return {"messages": [response]}
