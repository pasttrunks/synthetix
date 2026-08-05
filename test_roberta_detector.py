from transformers import pipeline

print("Loading local AI text detection model...")
pipe = pipeline("text-classification", model="fakespot-ai/roberta-base-ai-text-detection-v1")

text = "Schliemann's treatment of Priam's Treasure raises a serious cultural heritage issue. He presented the gold objects as one dramatic discovery connected to King Priam, although David Traill argues that the collection included items found at different times."

print("Running local AI detection classification...")
result = pipe(text)
print("Classification Result:", result)
