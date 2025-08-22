"""
Simple LangGraph Chatbot Example

This is a minimal example showing how to use the ConversationBot
for educational purposes in an AI course.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from aicourse.chatbot.conversation_bot import ConversationBot

load_dotenv()


def demo_basic_chat():
    """
    Demonstrate basic chatbot functionality.
    """
    print("🤖 Basic Chat Demo")
    print("=" * 40)

    # Create chatbot
    bot = ConversationBot(temperature=0.8)

    # Single interaction
    response, info = bot.chat("Hello! Can you explain what LangGraph is?")
    print(f"User: Hello! Can you explain what LangGraph is?")
    print(f"Bot:  {response}")
    print(f"📊 Info: {info}\n")


def demo_conversation_memory():
    """
    Demonstrate conversation memory and context retention.
    """
    print("🧠 Conversation Memory Demo")
    print("=" * 40)

    bot = ConversationBot()
    conversation_history = []

    # First turn
    print("Turn 1:")
    response1, _ = bot.chat(
        "I'm working on a Python project about machine learning.", conversation_history
    )
    conversation_history.extend(
        [
            HumanMessage(
                content="I'm working on a Python project about machine learning."
            ),
            AIMessage(content=response1),
        ]
    )
    print(f"User: I'm working on a Python project about machine learning.")
    print(f"Bot:  {response1}\n")

    # Second turn - references previous context
    print("Turn 2:")
    response2, info = bot.chat(
        "What libraries would you recommend for my project?", conversation_history
    )
    print(f"User: What libraries would you recommend for my project?")
    print(f"Bot:  {response2}")
    print(f"📊 Final info: {info}\n")


def demo_interactive_features():
    """
    Show different conversation features.
    """
    print("⚡ Interactive Features Demo")
    print("=" * 40)

    bot = ConversationBot(temperature=0.9)

    # Creative response
    response, _ = bot.chat("Write a haiku about artificial intelligence.")
    print("User: Write a haiku about artificial intelligence.")
    print(f"Bot:  {response}\n")

    # Technical explanation
    response, _ = bot.chat("Explain gradient descent in simple terms.")
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
        print("\nTo try interactive mode, run:")
        print("uv run python aicourse/chatbot/conversation_bot.py")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure your OPENAI_API_KEY is configured in .env")


if __name__ == "__main__":
    main()
