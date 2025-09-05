"""
Modern Conversational Chatbot using LangGraph StateGraph

This module demonstrates how to build a stateful chatbot using LangGraph,
showcasing conversation memory, state management, and graph-based workflows.
"""

import uuid
from typing import Optional, Iterator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.chatbot.graph import create_conversation_graph

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

    def chat_stream(
        self, user_input: str, thread_id: Optional[str] = None
    ) -> Iterator[str]:
        """
        Stream AI response tokens in real-time.

        Conversation history is automatically managed by the graph's checkpointer.

        Args:
            user_input: The user's message
            thread_id: Optional thread ID to maintain conversation context.
                      If None, creates a new conversation thread.

        Yields:
            Individual response tokens/chunks as they're generated
        """
        # Create thread ID if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Create config with thread ID for checkpointing
        config = {"configurable": {"thread_id": thread_id}}

        # Create input with just the new user message
        input_state = {"messages": [HumanMessage(content=user_input)]}

        # Stream the response using LangGraph's message streaming
        for chunk, metadata in self.graph.stream(
            input_state, config=config, stream_mode="messages"
        ):
            # Filter to only AI message chunks and yield content
            if isinstance(chunk, AIMessage) and chunk.content:
                yield chunk.content

    def start_conversation(self, streaming: bool = True) -> None:
        """
        Start an interactive conversation loop with persistent thread state.

        Args:
            streaming: Whether to stream responses in real-time (default: True)
        """
        mode_text = "Streaming" if streaming else "Standard"
        print(f"🤖 LangGraph Chatbot Ready! ({mode_text} Mode)")
        print("Type 'quit' to exit, 'new' to start new conversation\n")

        # Generate a thread ID for this conversation session
        thread_id = str(uuid.uuid4())
        print(f"📝 Conversation thread: {thread_id[:8]}...\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    # Let the graph handle the storage logic
                    ai_response = self.chat(user_input, thread_id)
                    if ai_response:  # Only print if there's a response
                        print(f"Bot: {ai_response}")
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "new":
                    thread_id = str(uuid.uuid4())
                    print(f"🆕 New conversation started! Thread: {thread_id[:8]}...")
                    continue

                if not user_input:
                    continue

                # Chat with persistent thread state
                if streaming:
                    # Stream the response in real-time
                    print("Bot: ", end="", flush=True)
                    for chunk in self.chat_stream(user_input, thread_id):
                        print(chunk, end="", flush=True)
                    print("\n")  # Add newline after streaming is complete
                else:
                    # Standard non-streaming response
                    ai_response = self.chat(user_input, thread_id)
                    print(f"Bot: {ai_response}\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue


def main():
    """
    Example usage of the ConversationBot with streaming responses.
    """
    bot = ConversationBot()

    # Start interactive session with streaming enabled by default
    bot.start_conversation(streaming=True)


if __name__ == "__main__":
    main()
