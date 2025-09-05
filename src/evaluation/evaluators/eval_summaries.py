import re

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import Client, evaluate
from langsmith.evaluation import LangChainStringEvaluator

from src.evaluation.youtube_summarizer import summarize_transcript

load_dotenv()
dataset_name = "youtube_summaries"
client = Client()


def call_chain(inputs: dict) -> str:
    return summarize_transcript(inputs["file_text"])


eval_llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
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
