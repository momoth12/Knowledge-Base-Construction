"""Tests for the KBC pipeline"""

from kbc.dataset import dataset_iterator
from kbc.model import LLM

it = dataset_iterator("train")
model = LLM("mistralai/Mistral-7B-Instruct-v0.3")

question = next(it)

prompt = f"Give me a bullet-point list of the names of the people who won the {question['SubjectEntity']}:"

print("prompt : ", prompt)

response = model.generate(prompt)

print("response : ", response)

# The output is something like this:
# - 1901: Emil von Behring
# - 1902: Ronald Ross
# - 1903: Niels Ryberg Finsen
# - 1904: Ivan Pav

# We would then need to query wikidata by names, identify the best matching ID, and submit it in the answer
