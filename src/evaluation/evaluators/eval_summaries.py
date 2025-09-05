import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate
from langsmith.evaluation import LangChainStringEvaluator

from src.evaluation.youtube_summarizer import summarize_transcript

load_dotenv()
dataset_name = "youtube_summaries"
client = Client()


def call_chain(inputs: dict) -> str:
    return summarize_transcript(inputs["file_text"])


eval_llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
criterion = {"creativity": "Is this submission creative and imaginative?"}
criteria_evaluator = LangChainStringEvaluator(
    "labeled_criteria", config={"criteria": criterion, "llm": eval_llm}
)


def contains_hashtags(outputs: str, reference_outputs: str) -> bool:
    found_hashtags = re.findall(r"#\w+", outputs["output"])
    return len(found_hashtags) >= 3


evaluate(
    call_chain,
    data=dataset_name,
    evaluators=[contains_hashtags, criteria_evaluator],
    experiment_prefix="BaseEval",
    max_concurrency=3,
    num_repetitions=1,
)
