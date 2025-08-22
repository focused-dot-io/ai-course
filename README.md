# AI Course Examples Repository

Welcome to the **Focused AI Course** - a comprehensive collection of practical examples and implementations for building AI-driven applications using modern Python tools and frameworks.

## 🚀 Overview

This repository contains hands-on examples covering key AI development concepts including:

- **Prompt Engineering**: Advanced techniques for optimizing AI prompts
- **Chatbot Development**: Building interactive conversational AI applications
- **Retrieval Augmented Generation (RAG)**: Building intelligent document processing systems
- **Audio Processing**: Transcription and analysis using Deepgram
- **Evaluation & Testing**: Systematic approaches to AI model evaluation
- **LangChain Integration**: Practical implementations using the LangChain ecosystem

## 🛠 Technology Stack

- **Python 3.11+** - Core programming language
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **[LangChain](https://langchain.readthedocs.io/)** - Framework for developing applications with LLMs
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Library for building stateful, multi-actor applications
- **[LangSmith](https://smith.langchain.com/)** - Platform for debugging, testing, and monitoring

## 📋 Prerequisites

Before getting started, you'll need to obtain API keys for the following services:

- **[OpenAI](https://platform.openai.com/api-keys)** - For GPT models
- **[Tavily](https://tavily.com/)** - For web search capabilities (free tier available)
- **[LangSmith](https://smith.langchain.com/)** - For tracing and monitoring
- **[Deepgram](https://deepgram.com/)** - For audio transcription

## 🔧 Installation

### Step 1: Install uv

If you don't have `uv` installed, choose your preferred installation method:

<details>
<summary><strong>Installation Options</strong></summary>

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

**Using Homebrew (macOS):**
```bash
brew install uv
```

</details>

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd aicourse

# Install dependencies
uv sync

# Copy the environment template (if .env doesn't exist)
cp .env.sample .env
```

### Step 3: Configure Environment

Edit the `.env` file with your API keys:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Tavily Search
TAVILY_API_KEY=your_tavily_api_key_here

# LangSmith Configuration
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="AICOURSE"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# Deepgram Configuration
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

## 🏃‍♂️ Usage

### Running Examples

Execute Python scripts with the project environment:

```bash
# Run a specific example
uv run python aicourse/prompt_engineering/few_shot_learning.py

# Run with module syntax
uv run python -m aicourse.evaluation.youtube_summarizer
```

### Interactive Development

Activate the project environment for interactive work:

```bash
# Activate shell with project dependencies
uv shell

# Now you can run Python directly
python aicourse/prompt_engineering/advertising_costar.py
```

## 📁 Project Structure

```
aicourse/
├── chatbot/             # Interactive chatbot implementations
├── evaluation/          # Model evaluation and testing examples
│   ├── evaluators/      # Custom evaluation functions
│   ├── transcription/   # Audio processing utilities
│   └── transcripts/     # Sample transcript files
└── prompt_engineering/  # Prompt optimization examples
    └── prompts/         # Structured prompt templates
```

## 🔄 Development Workflow

### Managing Dependencies

```bash
# Add a new package
uv add package-name

# Add development dependency
uv add --dev package-name

# Remove a package
uv remove package-name

# Update all dependencies
uv sync --upgrade

# List installed packages
uv pip list
```

### Running Tests and Evaluations

```bash
# Run evaluation examples
uv run python aicourse/evaluation/evaluators/eval_summaries.py

# Run prompt engineering examples
uv run python aicourse/prompt_engineering/national_parks.py

# Run chatbot examples
uv run python aicourse/chatbot/your_chatbot_script.py
```

## 📚 Learning Path

1. **Start with Prompt Engineering** - Learn fundamental techniques for crafting effective prompts
2. **Build Interactive Chatbots** - Create conversational AI applications with real-time interactions
3. **Explore Evaluation Methods** - Understand how to measure and improve AI performance
4. **Build RAG Applications** - Create document-based question-answering systems
5. **Advanced Integration** - Combine multiple AI services for complex workflows

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [LangChain Documentation](https://langchain.readthedocs.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

---

**Happy Learning!** 🎓 If you have questions or run into issues, please open an issue in this repository.