"""
Simple LangGraph Chatbot Example

This is a minimal example showing how to use the ConversationBot
for educational purposes in an AI course.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.chatbot.conversation_bot import ConversationBot

load_dotenv()


def demo_basic_chat():
    """
    Demonstrate basic chatbot functionality.
    """
    print("🤖 Basic Chat Demo")
    print("=" * 40)

    # Create chatbot
    bot = ConversationBot()

    # Single interaction
    response = bot.chat("Hello! Can you explain what LangGraph is?")
    print(f"User: Hello! Can you explain what LangGraph is?")
    print(f"Bot:  {response}\n")


def demo_conversation_memory():
    """
    Demonstrate automatic conversation memory via LangGraph checkpointer.
    """
    print("🧠 Conversation Memory Demo")
    print("=" * 40)

    bot = ConversationBot()
    thread_id = "ml-project-conversation"

    # First turn
    print("Turn 1:")
    response1 = bot.chat(
        "I'm working on a Python project about machine learning.", thread_id
    )
    print(f"User: I'm working on a Python project about machine learning.")
    print(f"Bot:  {response1}\n")

    # Second turn - automatically remembers previous context via checkpointer
    print("Turn 2:")
    response2 = bot.chat(
        "What libraries would you recommend for my project?", thread_id
    )
    print(f"User: What libraries would you recommend for my project?")
    print(f"Bot:  {response2}\n")


def demo_interactive_features():
    """
    Show different conversation features.
    """
    print("⚡ Interactive Features Demo")
    print("=" * 40)

    bot = ConversationBot()

    # Creative response
    response = bot.chat("Write a haiku about artificial intelligence.")
    print("User: Write a haiku about artificial intelligence.")
    print(f"Bot:  {response}\n")

    # Technical explanation
    response = bot.chat("Explain gradient descent in simple terms.")
    print("User: Explain gradient descent in simple terms.")
    print(f"Bot:  {response}\n")


def main():
    """
    Run all chatbot examples.
    """
    print("🚀 LangGraph Chatbot Examples")
    print("=" * 50)
    print()

    try:
        # Run demos
        demo_basic_chat()
        demo_conversation_memory()
        demo_interactive_features()

        print("✅ All demos completed successfully!")
        print("\n🚀 Interactive Options:")
        print("Streaming mode:     uv run python src/chatbot/conversation_bot.py")
        print(
            "Non-streaming mode: uv run python -c 'from src.chatbot.conversation_bot import ConversationBot; ConversationBot().start_conversation(streaming=False)'"
        )

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure your OPENAI_API_KEY is configured in .env")


if __name__ == "__main__":
    main()
