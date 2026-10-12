
from pathlib import Path
import json
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "results" / "live_model"

print("Loading saved SCRIPT-CERT model...")

with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

classifier = joblib.load(MODEL_DIR / "classifier.joblib")
encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

ENCODER_DIR = MODEL_DIR / "encoder"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"

tokenizer = AutoTokenizer.from_pretrained(
    str(TOKENIZER_DIR),
    use_fast=False,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    str(ENCODER_DIR),
    local_files_only=True
)
model.eval()

device = torch.device("cpu")
model.to(device)

max_length = int(config.get("max_length", 64))

def embed_comment(text):
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embedding = (output.last_hidden_state * mask).sum(dim=1)
        embedding /= mask.sum(dim=1).clamp(min=1e-9)

    return embedding.cpu().numpy()

print("\nSCRIPT-CERT live comment tester")
print("Type 'exit' to quit.")

while True:
    comment = input("\nEnter a Telugu-English comment: ").strip()

    if comment.lower() == "exit":
        break

    if not comment:
        print("Please enter a comment.")
        continue

    vector = embed_comment(comment)
    probabilities = classifier.predict_proba(vector)[0]
    prediction = classifier.predict(vector)[0]
    label = encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))

    print("\nPrediction:", label)
    print(f"Model confidence: {confidence:.2%}")

    if confidence < 0.70:
        print("Recommendation: Human review")
    else:
        print("Recommendation: Review still advised for real-world moderation")
