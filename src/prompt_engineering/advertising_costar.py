from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

from src.prompt_engineering.prompts.advertising.costar_prompt_ads import (
    costar_prompt_advertising,
)
from src.prompt_engineering.prompts.advertising.generic_prompt_ads import (
    generic_prompt_advertising,
)

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)


def run_costar_examples():
    GENERIC_PROMPT = ChatPromptTemplate.from_template(generic_prompt_advertising)
    COSTAR_PROMPT = ChatPromptTemplate.from_template(costar_prompt_advertising)
    product = "sneakers"
    target_audience = "Millennials"

    print("====================================")
    print("GENERIC PROMPT")
    print("====================================")

    run_rag_chain(GENERIC_PROMPT, product, target_audience)

    print("====================================")
    print("COSTAR PROMPT")
    print("====================================")
    run_rag_chain(COSTAR_PROMPT, product, target_audience)


def run_rag_chain(prompt, product, target_audience):
    rag_chain = (
        RunnableParallel(
            product=itemgetter("product"),
            target_audience=itemgetter("target_audience"),
        )
        | prompt
        | llm
    )

    result = rag_chain.invoke({"product": product, "target_audience": target_audience})

    print("ANSWER")
    print("====================================")
    print(result.content)


if __name__ == "__main__":
    run_costar_examples()
