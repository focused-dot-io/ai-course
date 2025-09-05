"""
Response generation node for the conversation chatbot.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState


def generate_response(state: MessagesState) -> MessagesState:
    """
    Generate AI response based on conversation history.

    Args:
        state: Current conversation state with message history

    Returns:
        Updated state with AI response added
    """
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

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
