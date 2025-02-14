from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langsmith import traceable

from aicourse.prompt_engineering.prompts.national_parks.costar_prompt import costar_prompt
from aicourse.prompt_engineering.prompts.national_parks.generic_prompt import generic_prompt

load_dotenv()
llm = ChatOpenAI(
    temperature=0,
    model='gpt-4-1106-preview',
    streaming=True
)


def run_costar_examples():
    GENERIC_PROMPT = ChatPromptTemplate.from_template(generic_prompt)
    COSTAR_PROMPT = ChatPromptTemplate.from_template(costar_prompt)
    destination = "Rocky Mountain National Park"
    season = "summer"
    duration = "3 days"
    interests = "hiking, wildlife watching"

    context = get_internet_articles(
        destination,
        season,
        duration,
        interests
    )
    print("GENERIC PROMPT")
    print("====================================")

    result = run_rag_chain(GENERIC_PROMPT, context, destination, season, duration, interests)

    print("ANSWER")
    print("====================================")
    print(result["answer"].content)

    print("DOCS")
    print("====================================")
    print(result["docs"])

    print("COSTAR PROMPT")
    print("====================================")
    result = run_rag_chain(COSTAR_PROMPT, context, destination, season, duration, interests)

    print("ANSWER")
    print("====================================")
    print(result["answer"].content)

    print("DOCS")
    print("====================================")
    print(result["docs"])


def get_internet_articles(destination: str, season: str, duration: str, interests: str) -> []:
    question = f"Planning a trip to {destination} in {season} for {duration} with interests in {interests}"

    web_docs = TavilySearchResults(k=5).invoke({"query": question})
    web_results = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in web_docs]

    return web_results


@traceable(name="national parks")
def run_rag_chain(prompt, context, destination, season, duration, interests):
    rag_chain = (RunnableParallel(
        context=itemgetter("context"),
        destination=itemgetter("destination"),
        duration=itemgetter("duration"),
        season=itemgetter("season"),
        interests=itemgetter("interests")
    ) | RunnableParallel(
        answer=(prompt | llm),
        docs=itemgetter("context")
    ))

    return rag_chain.invoke({
        "context": context,
        "destination": destination,
        "season": season,
        "duration": duration,
        "interests": interests
    })


if __name__ == "__main__":
    run_costar_examples()
