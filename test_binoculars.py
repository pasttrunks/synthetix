import sys
import torch
from binoculars import Binoculars

bino = Binoculars()

text = "Schliemann's treatment of Priam's Treasure raises a serious cultural heritage issue. He presented the gold objects as one dramatic discovery connected to King Priam."

print("Running Binoculars prediction...")
score = bino.predict(text)
print("Binoculars score:", score)
