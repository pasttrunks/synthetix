import json
from transformers import pipeline

with open("full_troy_essay.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    essay = data["query_text"]

print("--- Model 1: OpenAI RoBERTa Detector ---")
pipe1 = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")
res1 = pipe1(essay[:500])
print("OpenAI RoBERTa Result:", res1)

print("--- Model 2: FakeSpot RoBERTa Detector ---")
pipe2 = pipeline("text-classification", model="fakespot-ai/roberta-base-ai-text-detection-v1")
res2 = pipe2(essay[:500])
print("FakeSpot Result:", res2)
