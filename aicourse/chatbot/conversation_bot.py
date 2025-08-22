"""
Modern Conversational Chatbot using LangGraph StateGraph

This module demonstrates how to build a stateful chatbot using LangGraph,
showcasing conversation memory, state management, and graph-based workflows.
"""

from typing import Annotated, List, TypedDict
import operator
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


class ConversationState(TypedDict):
    """
    State schema for the conversation chatbot.

    This defines the structure of data that flows through the graph nodes.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    conversation_count: Annotated[int, operator.add]


class ConversationBot:
    """
    A conversational chatbot powered by LangGraph StateGraph.

    Features:
    - Maintains conversation history
    - Tracks conversation statistics
    - Uses modern LangGraph patterns
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.system_prompt = ChatPromptTemplate.from_messages(
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

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for conversation flow.

        The graph defines how messages flow and are processed.
        """
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("respond", self._generate_response)
        workflow.add_node("update_counter", self._update_conversation_count)

        # Define the flow
        workflow.add_edge(START, "respond")
        workflow.add_edge("respond", "update_counter")
        workflow.add_edge("update_counter", END)

        return workflow.compile()

    def _generate_response(self, state: ConversationState) -> ConversationState:
        """
        Generate AI response based on conversation history.

        Args:
            state: Current conversation state with message history

        Returns:
            Updated state with AI response added
        """
        # Create the chain
        chain = self.system_prompt | self.llm

        # Generate response
        response = chain.invoke({"messages": state["messages"]})

        # Return updated state
        return {
            "messages": [response],
            "conversation_count": 0,  # Will be updated in next node
        }

    def _update_conversation_count(self, state: ConversationState) -> ConversationState:
        """
        Update the conversation turn counter.

        Args:
            state: Current conversation state

        Returns:
            Updated state with incremented counter
        """
        return {
            "messages": [],  # No new messages to add
            "conversation_count": 1,  # Increment by 1
        }

    def chat(
        self, user_input: str, conversation_history: List[BaseMessage] = None
    ) -> tuple[str, dict]:
        """
        Process a user input and return the AI response.

        Args:
            user_input: The user's message
            conversation_history: Previous messages (optional)

        Returns:
            Tuple of (AI response text, updated state info)
        """
        if conversation_history is None:
            conversation_history = []

        # Add user message to history
        current_messages = conversation_history + [HumanMessage(content=user_input)]

        # Create initial state
        initial_state = {"messages": current_messages, "conversation_count": 0}

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Extract AI response
        ai_message = result["messages"][-1]
        ai_response = (
            ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        )

        # Return response and state info
        state_info = {
            "total_messages": len(result["messages"]),
            "conversation_turns": result["conversation_count"],
        }

        return ai_response, state_info

    def start_conversation(self) -> None:
        """
        Start an interactive conversation loop.

        This demonstrates how to use the chatbot in a continuous conversation.
        """
        print("🤖 LangGraph Chatbot Ready!")
        print("Type 'quit' to exit, 'clear' to clear history\n")

        conversation_history = []
        total_turns = 0

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "clear":
                    conversation_history = []
                    total_turns = 0
                    print("🧹 Conversation history cleared!")
                    continue

                if not user_input:
                    continue

                # Get AI response
                ai_response, state_info = self.chat(user_input, conversation_history)

                # Update conversation history
                conversation_history.extend(
                    [HumanMessage(content=user_input), AIMessage(content=ai_response)]
                )

                total_turns += 1

                # Display response
                print(f"Bot: {ai_response}")
                print(
                    f"💬 Turn {total_turns} | Messages: {len(conversation_history)}\n"
                )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue


def main():
    """
    Example usage of the ConversationBot.
    """
    # Create chatbot instance
    bot = ConversationBot()

    # Example single interaction
    print("=== Single Chat Example ===")
    response, info = bot.chat("Hello! What can you help me with?")
    print(f"AI: {response}")
    print(f"State: {info}\n")

    # Example with conversation history
    print("=== Conversation History Example ===")
    history = []

    # First message
    response1, _ = bot.chat("My name is Alice and I love Python programming.", history)
    history.extend(
        [
            HumanMessage(content="My name is Alice and I love Python programming."),
            AIMessage(content=response1),
        ]
    )
    print(f"AI: {response1}")

    # Second message with context
    response2, info2 = bot.chat("What's my name and what do I like?", history)
    print(f"AI: {response2}")
    print(f"State: {info2}\n")

    # Start interactive mode
    print("=== Interactive Mode ===")
    bot.start_conversation()


if __name__ == "__main__":
    main()
