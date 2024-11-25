from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    temperature=0,
    model='gpt-4-1106-preview',
    streaming=True
)


def run_example():
    # TODO: Update the prompt with a new example.
    prompt = """
    # Instructions
    =================
    Given an animal and its description, extract the animal's attributes in json format.
    
    #Examples
    =================
    ## Example 1
    Animal Description: The African elephant is the largest land animal on Earth. It is known for its large ears, tusks, and trunk.
    Animal Attributes: {{
      "name": "African elephant",
      "biological_class": "mammalia",
      "notable_features": [
        "large ears",
        "tusks",
        "trunk"
      ]
    }}

    ## Example 2
    
    # Begin Analysis
    =================
    Animal Description: {animal_description}
    Animal Attributes: 
    """

    PROMPT_TEMPLATE = ChatPromptTemplate.from_template(prompt)
    animal_description = "The domestic cat is a small, typically furry, carnivorous mammal. They are often called house cats when kept as indoor pets or simply cats when there is no need to distinguish them from other felids and felines."

    rag_chain = RunnablePassthrough(
        animal_description=itemgetter("animal_description"),
    ) | PROMPT_TEMPLATE | llm

    result = rag_chain.invoke({
        "animal_description": animal_description,
    })

    print("ANSWER")
    print("====================================")
    print(result.content)


if __name__ == "__main__":
    run_example()
