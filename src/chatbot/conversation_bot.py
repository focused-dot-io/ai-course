"""
Modern Conversational Chatbot using LangGraph StateGraph

This module demonstrates how to build a stateful chatbot using LangGraph,
showcasing conversation memory, state management, and graph-based workflows.
"""

import uuid
from typing import Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from .graph import create_conversation_graph

load_dotenv()


class ConversationBot:
    """
    A conversational chatbot powered by LangGraph StateGraph with checkpointing.

    Features:
    - Automatic conversation memory via LangGraph checkpointer
    - Thread-based conversation management
    - LangGraph Studio compatible structure
    """

    def __init__(self):
        self.graph = create_conversation_graph()

    def chat(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """
        Process a user input and return the AI response.

        Conversation history is automatically managed by the graph's checkpointer.

        Args:
            user_input: The user's message
            thread_id: Optional thread ID to maintain conversation context.
                      If None, creates a new conversation thread.

        Returns:
            AI response text
        """
        # Create thread ID if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Create config with thread ID for checkpointing
        config = {"configurable": {"thread_id": thread_id}}

        # Create input with just the new user message
        # The graph's checkpointer handles conversation history automatically
        input_state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph - it automatically loads previous state and saves new state
        result = self.graph.invoke(input_state, config=config)

        # Extract AI response
        ai_message = result["messages"][-1]
        ai_response = (
            ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        )

        return ai_response

    def start_conversation(self) -> None:
        """
        Start an interactive conversation loop with persistent thread state.
        """
        print("🤖 LangGraph Chatbot Ready!")
        print("Type 'quit' to exit, 'new' to start new conversation\n")

        # Generate a thread ID for this conversation session
        thread_id = str(uuid.uuid4())
        print(f"📝 Conversation thread: {thread_id[:8]}...\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "new":
                    thread_id = str(uuid.uuid4())
                    print(f"🆕 New conversation started! Thread: {thread_id[:8]}...")
                    continue

                if not user_input:
                    continue

                # Chat with persistent thread state
                ai_response = self.chat(user_input, thread_id)

                # Display response
                print(f"Bot: {ai_response}\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue


def main():
    """
    Example usage of the ConversationBot with thread-based state management.
    """
    bot = ConversationBot()

    bot.start_conversation()


if __name__ == "__main__":
    main()
