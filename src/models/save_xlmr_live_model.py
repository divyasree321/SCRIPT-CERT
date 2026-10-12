
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "train_clean.csv"
OUTPUT_DIR = ROOT / "results" / "live_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
print("Loading training data...")

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
df["text"] = df["text"].fillna("").astype(str)
df["label"] = df["label"].astype(str)

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

print("Training rows:", len(df))
print("Labels:", list(encoder.classes_))
print("Loading XLM-R tokenizer and encoder...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

def get_embeddings(texts):
    batches = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            batches.append(pooled.cpu().numpy())

            print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)}")

    return np.vstack(batches)

print("Generating XLM-R embeddings. This may take time on CPU...")
X = get_embeddings(df["text"].tolist())

print("Training Logistic Regression...")
classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
classifier.fit(X, y)

joblib.dump(classifier, OUTPUT_DIR / "classifier.joblib")
joblib.dump(encoder, OUTPUT_DIR / "label_encoder.joblib")
tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
model.save_pretrained(OUTPUT_DIR / "encoder")

config = {
    "base_model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "pooling": "attention-mask-weighted mean pooling",
    "classifier": "LogisticRegression",
    "training_rows": len(df),
    "labels": list(encoder.classes_),
    "device_used_for_creation": str(DEVICE),
}

(OUTPUT_DIR / "config.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

print("\nSaved live model to:", OUTPUT_DIR)
print("Done. You do not need to repeat this step unless the training data changes.")
