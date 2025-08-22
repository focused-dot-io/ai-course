from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)


def run_example():
    # TODO: Update the prompt to output xml.
    prompt = """
    # Instructions
    =================
    Recommend the 5 best {genre} music albums from {year}. 
    
    
    # Output Format
    =================
    Create an array of json objects. Each object should be formatted as follows:
    {{
      'artist':'Fleetwood Mac',
      'album_name':'Rumors'
    }}
    """
    PROMPT_TEMPLATE = ChatPromptTemplate.from_template(prompt)
    genre = "rock"
    year = "2021"

    rag_chain = (
        RunnableParallel(
            genre=itemgetter("genre"),
            year=itemgetter("year"),
        )
        | PROMPT_TEMPLATE
        | llm
    )

    result = rag_chain.invoke({"genre": genre, "year": year})

    print("ANSWER")
    print("====================================")
    print(result.content)


if __name__ == "__main__":
    run_example()
