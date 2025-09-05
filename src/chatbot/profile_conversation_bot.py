#!/usr/bin/env python3
"""
Interactive profile-based conversation bot using LangGraph Store.
Shows how user profiles are built and used over time.
"""

import uuid
from typing import Optional
from langchain_core.messages import HumanMessage

from src.chatbot.graph.profile_graph import create_profile_graph


class ProfileConversationBot:
    """
    Profile-based conversational AI bot using LangGraph Store.
    Builds and maintains user profiles instead of storing conversation summaries.
    """

    def __init__(self):
        """Initialize the profile conversation bot."""
        self.graph = create_profile_graph()
        print("🤖 Profile Chatbot initialized with LangGraph Store!")

    def chat(self, message: str, thread_id: str) -> str:
        """
        Process a single message and return AI response with streaming.

        Args:
            message: User's message
            thread_id: Conversation thread identifier

        Returns:
            AI response string
        """
        try:
            # Create input state
            input_state = {"messages": [HumanMessage(content=message)]}
            config = {"configurable": {"thread_id": thread_id}}

            # Stream the response
            response_content = ""
            print("Bot: ", end="", flush=True)

            # Try streaming with different approach
            for chunk in self.graph.stream(input_state, config=config):
                # Check if chunk contains messages from any node
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        if isinstance(node_output, dict) and "messages" in node_output:
                            for message in node_output["messages"]:
                                if hasattr(message, "type") and (
                                    message.type == "ai" or message.type == "system"
                                ):
                                    content = message.content
                                    print(content, end="", flush=True)
                                    response_content = content
                                    break

            print()  # New line after response
            return (
                response_content
                if response_content
                else "I apologize, but I couldn't generate a response."
            )

        except Exception as e:
            print(f"❌ Error processing message: {e}")
            return "I encountered an error processing your message."

    def start_conversation(self):
        """Start an interactive conversation session."""
        print("\n🤖 Profile Chatbot Ready! (LangGraph Store)")
        print("Type 'quit' to exit, 'new' to start new conversation")
        print("Watch how your profile builds over time!")

        thread_id = str(uuid.uuid4())[:8]
        print(f"\n📝 Conversation thread: {thread_id}...")

        try:
            while True:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() == "quit":
                    # Store final profile update
                    response = self.chat("quit", thread_id)
                    print(f"Bot: {response}")
                    break
                elif user_input.lower() == "new":
                    thread_id = str(uuid.uuid4())[:8]
                    print(f"\n📝 New conversation thread: {thread_id}...")
                    continue

                # Process message and get streaming response
                response = self.chat(user_input, thread_id)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n❌ Conversation error: {e}")

        print("\n👋 Goodbye!")


if __name__ == "__main__":
    """Run the interactive profile chatbot."""
    try:
        bot = ProfileConversationBot()
        bot.start_conversation()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print("💡 Make sure all dependencies are installed: uv sync")
