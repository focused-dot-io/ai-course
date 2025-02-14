import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langsmith import traceable

from aicourse.evaluation.prompts.summarize_transcript import summarize_transcript_prompt

load_dotenv()
llm = ChatOpenAI(
    temperature=0,
    model='gpt-4-1106-preview',
)

files = [
    "Chat With Your PDFs： Part 1 - An End to End LangChain Tutorial.txt",
    "Chat With Your PDFs： Part 2 - Frontend - An End to End LangChain Tutorial.txt",
    "Unlock Advanced RAG Secrets： LangChain Tutorial Finale.txt",
    "Unlock the Power of LangChain： Deploying to Production Made Easy.txt"
]


def get_file_contents(file_name):
    file_path = os.path.join(os.path.dirname(__file__), f"./transcripts/{file_name}")

    with open(file_path, "r") as file:
        return file.read()


@traceable
def summarize_transcript(file_text):
    PROMPT_TEMPLATE = ChatPromptTemplate.from_template(summarize_transcript_prompt)
    rag_chain = RunnablePassthrough(
        transcript=itemgetter("transcript")
    ) | PROMPT_TEMPLATE | llm

    return rag_chain.invoke({
        "transcript": file_text,
    })


def main():
    # for file in files:
    file_text = get_file_contents(files[0])
    summary = summarize_transcript(file_text)
    print(summary)

main()
