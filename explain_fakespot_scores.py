import json
from transformers import pipeline

pipe = pipeline("text-classification", model="fakespot-ai/roberta-base-ai-text-detection-v1")

with open("full_troy_essay.json", "r", encoding="utf-8") as f:
    essay = json.load(f)["query_text"]

paragraphs = essay.split("\n\n")

print("--- Sentence / Paragraph Breakdown ---")
for i, p in enumerate(paragraphs, 1):
    p_clean = p.strip()
    if not p_clean:
        continue
    res = pipe(p_clean[:512])[0]
    label = res["label"]
    score = res["score"]
    print(f"Paragraph {i}: [{label} {score*100:.2f}%]")
    print(f"Text snippet: \"{p_clean[:100]}...\"\n")
