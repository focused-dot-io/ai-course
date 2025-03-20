This is the repository for examples on the FocusedLabs AI RAG course.

# Pre-requisites:
- Obtain an OpenAI API key
- Obtain a Tavily API key (free)
- Sign up for Langsmith and get an API key
- Obtain a DeepGram API key

# Setup:
- Install any dependencies with `poetry install`
- Create a .env file with the following variables:
```
OPENAI_API_KEY=<Open AI API key>
TAVILY_API_KEY=<Your Tavily key>
LANGCHAIN_API_KEY=<Your Langsmith/Langchain api key>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="AICOURSE"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
DEEPGRAM_API_KEY=<Your Deepgram API key>
```

