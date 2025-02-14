from dotenv import load_dotenv
from langsmith import Client

example_inputs = [
    ("What is the largest mammal?", "The blue whale"),
    ("What do mammals and birds have in common?", "They are both warm-blooded"),
    ("What are reptiles known for?", "Having scales"),
    ("What's the main characteristic of amphibians?", "They live both in water and on land"),
]
load_dotenv()
client = Client()
dataset_name = "Elementary Animal Questions"

# Storing inputs in a dataset lets us
# run chains and LLMs over a shared set of examples.
dataset = client.create_dataset(
    dataset_name=dataset_name, description="Questions and answers about animal phylogenetics.",
)

# Prepare inputs, outputs, and metadata for bulk creation
inputs = [{"question": input_prompt} for input_prompt, _ in example_inputs]
outputs = [{"answer": output_answer} for _, output_answer in example_inputs]
metadata = [{"source": "Wikipedia"} for _ in example_inputs]

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    metadata=metadata,
    dataset_id=dataset.id,
)