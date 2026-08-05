from transformers import pipeline

pipe = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")

text = "Schliemann's treatment of Priam's Treasure raises a serious cultural heritage issue. He presented the gold objects as one dramatic discovery connected to King Priam, although David Traill argues that the collection included items found at different times."

result = pipe(text)
print("Classification Result:", result)
