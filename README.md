# AI Course Examples Repository

Welcome to the **Focused AI Course** - a comprehensive collection of practical examples and implementations for building AI-driven applications using modern Python tools and frameworks.

## 🚀 Overview

Practical examples covering key AI development concepts:

- **Prompt Engineering**: Optimizing AI prompts with COSTAR framework
- **Chatbot Development**: LangGraph-based conversational AI
- **RAG Applications**: Document processing with web search
- **Audio Processing**: Transcription and summarization
- **Evaluation**: Model testing and performance measurement

## 🛠 Tech Stack

- **Python 3.11+** with **uv** package manager
- **LangChain & LangGraph** for AI applications
- **OpenAI GPT-4o** for language models
- **LangSmith** for monitoring and evaluation

## 📋 Prerequisites

API keys needed:
- **OpenAI** - GPT models
- **Tavily** - Web search (free tier)
- **LangSmith** - Monitoring
- **Deepgram** - Audio transcription

## 🔧 Setup

1. **Install uv**: `pip install uv` (or see [uv docs](https://docs.astral.sh/uv/))
2. **Install dependencies**: `uv sync`  
3. **Configure API keys**: Copy `.env.sample` to `.env` and add your keys

## 🏃‍♂️ Usage

Run examples with: `uv run python aicourse/module/script.py`

## 📁 Project Structure

```
aicourse/
├── chatbot/             # LangGraph conversational AI
├── evaluation/          # Model evaluation and testing  
└── prompt_engineering/  # Prompt optimization
```

## 📚 Examples

```bash
# Prompt engineering with web search
uv run python aicourse/prompt_engineering/national_parks.py

# Interactive chatbot  
uv run python aicourse/chatbot/conversation_bot.py

# Evaluation examples
uv run python aicourse/evaluation/evaluators/eval_summaries.py
```

## 🎓 Learning Path

1. **Prompt Engineering** - Effective prompting techniques
2. **Chatbots** - Conversational AI with LangGraph  
3. **Evaluation** - Testing and measuring AI performance
4. **RAG Applications** - Document-based Q&A systems