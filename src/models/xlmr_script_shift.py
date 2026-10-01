"""
SCRIPT-CERT
XLM-RoBERTa frozen-embedding baseline under SCRIPT-SHIFT conditions.

Model:
    xlm-roberta-base

Approach:
    1. Load XLM-RoBERTa
    2. Extract frozen CLS-style sentence embeddings
    3. Train Logistic Regression on original training data
    4. Evaluate on:
       - original
       - romanized
       - telugu_script
       - mixed_script
       - mild_noise
       - severe_noise

No fine-tuning is performed.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MODEL_NAME = "xlm-roberta-base"

TRAIN_FILE = "data/processed/train_clean.csv"
SCRIPT_SHIFT_FILE = "data/script_shift/validation_script_shift_full.csv"

PREDICTION_OUTPUT = "results/predictions/xlmr_script_shift_predictions.csv"
RESULTS_OUTPUT = "results/tables/xlmr_script_shift_results.csv"
FLIP_OUTPUT = "results/tables/xlmr_prediction_flip_rates.csv"
CONFIDENCE_OUTPUT = "results/tables/xlmr_confidence_shift.csv"
TRAINING_INFO_OUTPUT = "results/tables/xlmr_training_info.csv"
CONFIG_OUTPUT = "results/tables/xlmr_config.json"

MAX_LENGTH = 128
BATCH_SIZE = 8

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("SCRIPT-CERT - XLM-R SCRIPT-SHIFT EXPERIMENT")
print("=" * 80)

print(f"\nModel       : {MODEL_NAME}")
print(f"Device      : {DEVICE}")
print(f"Max length  : {MAX_LENGTH}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Random seed : {SEED}")

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("results/predictions", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
shift_df = pd.read_csv(SCRIPT_SHIFT_FILE, encoding="utf-8-sig")

print(f"\nTraining data shape     : {train_df.shape}")
print(f"Script-shift data shape: {shift_df.shape}")

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nScript-shift columns:")
print(shift_df.columns.tolist())

# ============================================================
# STANDARDIZE TRAINING DATA
# ============================================================

if "text" not in train_df.columns:
    if "Comments" in train_df.columns:
        train_df["text"] = train_df["Comments"]

if "label" not in train_df.columns:
    if "Label" in train_df.columns:
        train_df["label"] = train_df["Label"]

train_df["text"] = train_df["text"].fillna("").astype(str)
train_df["label"] = train_df["label"].astype(str)

shift_df["original_text"] = shift_df["original_text"].fillna("").astype(str)
shift_df["label"] = shift_df["label"].astype(str)

# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train_df["label"])

print("\nLabel mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i} -> {label}")

print("\nTraining label distribution:")
print(train_df["label"].value_counts())

# ============================================================
# LOAD XLM-R
# ============================================================

print("\n" + "=" * 80)
print("LOADING XLM-RoBERTa")
print("=" * 80)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

# Freeze model
for param in model.parameters():
    param.requires_grad = False

print("\nXLM-R loaded successfully.")

# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def get_embeddings(texts):
    """
    Extract frozen XLM-R sentence embeddings.

    Uses mean pooling over the last hidden states,
    weighted by the attention mask.
    """

    all_embeddings = []

    total = len(texts)

    print(f"\nGenerating embeddings for {total} texts...")

    with torch.no_grad():

        for start in range(0, total, BATCH_SIZE):

            batch_texts = texts[start:start + BATCH_SIZE]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            outputs = model(**encoded)

            hidden_states = outputs.last_hidden_state

            attention_mask = encoded["attention_mask"]

            mask = attention_mask.unsqueeze(-1).expand(
                hidden_states.size()
            ).float()

            masked_embeddings = hidden_states * mask

            summed = masked_embeddings.sum(dim=1)

            counts = mask.sum(dim=1).clamp(min=1e-9)

            embeddings = summed / counts

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

            processed = min(start + BATCH_SIZE, total)

            if processed % 100 == 0 or processed == total:
                print(
                    f"  Processed {processed}/{total}"
                )

    return np.vstack(all_embeddings)


# ============================================================
# TRAINING EMBEDDINGS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING TRAINING EMBEDDINGS")
print("=" * 80)

train_texts = train_df["text"].tolist()

X_train_embeddings = get_embeddings(train_texts)

print(
    f"\nTraining embedding shape: "
    f"{X_train_embeddings.shape}"
)

# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 80)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 80)

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=SEED
)

classifier.fit(
    X_train_embeddings,
    y_train
)

print("\nLogistic Regression training complete.")

# ============================================================
# SCRIPT-SHIFT CONDITIONS
# ============================================================

conditions = [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise",
]

# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
summary_results = []

original_predictions = None
original_confidence = None

for condition in conditions:

    print("\n" + "=" * 80)
    print(f"EVALUATING: {condition.upper()}")
    print("=" * 80)

    texts = shift_df[condition].fillna("").astype(str).tolist()

    y_true = label_encoder.transform(
        shift_df["label"].astype(str)
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = get_embeddings(texts)

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = classifier.predict(embeddings)

    probabilities = classifier.predict_proba(embeddings)

    confidence = probabilities.max(axis=1)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        predictions
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    mean_confidence = confidence.mean()

    print(f"\nAccuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"Macro Precision    : {macro_precision:.4f}")
    print(f"Macro Recall       : {macro_recall:.4f}")
    print(f"Mean Confidence    : {mean_confidence:.4f}")

    # --------------------------------------------------------
    # Store original predictions
    # --------------------------------------------------------

    if condition == "original":

        original_predictions = predictions.copy()
        original_confidence = confidence.copy()

    # --------------------------------------------------------
    # Flip rate
    # --------------------------------------------------------

    if original_predictions is not None:

        flip_rate = np.mean(
            predictions != original_predictions
        )

    else:

        flip_rate = 0.0

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    summary_results.append({
        "Condition": condition,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Macro-F1": macro_f1,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Mean Confidence": mean_confidence,
        "Prediction Flip Rate": flip_rate
    })

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    for i in range(len(texts)):

        all_predictions.append({
            "id": shift_df.iloc[i]["id"],
            "true_label": shift_df.iloc[i]["label"],
            "condition": condition,
            "text": texts[i],
            "prediction": label_encoder.inverse_transform(
                [predictions[i]]
            )[0],
            "confidence": float(confidence[i]),
            "probability_hate": float(
                probabilities[i][
                    list(label_encoder.classes_).index("hate")
                ]
            ) if "hate" in label_encoder.classes_ else np.nan,
            "probability_non_hate": float(
                probabilities[i][
                    list(label_encoder.classes_).index("non-hate")
                ]
            ) if "non-hate" in label_encoder.classes_ else np.nan
        })

# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(summary_results)

print("\n" + "=" * 80)
print("XLM-R RESULTS")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

# ============================================================
# SAVE MAIN RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_OUTPUT,
    index=False
)

print(
    f"\nSaved results to:\n{RESULTS_OUTPUT}"
)

# ============================================================
# PREDICTION FLIP RATES
# ============================================================

flip_rows = []

for condition in conditions:

    if condition == "original":
        flip_rate = 0.0
    else:

        condition_predictions = label_encoder.transform(
            shift_df["label"].astype(str)
        )

        condition_texts = shift_df[
            condition
        ].fillna("").astype(str).tolist()

        condition_embeddings = get_embeddings(
            condition_texts
        )

        condition_predictions = classifier.predict(
            condition_embeddings
        )

        flip_rate = np.mean(
            condition_predictions != original_predictions
        )

    flip_rows.append({
        "condition": condition,
        "flip_rate": flip_rate,
        "flip_count": int(
            round(flip_rate * len(shift_df))
        ),
        "total_samples": len(shift_df)
    })

flip_df = pd.DataFrame(flip_rows)

flip_df.to_csv(
    FLIP_OUTPUT,
    index=False
)

print(
    f"\nSaved flip rates to:\n{FLIP_OUTPUT}"
)

# ============================================================
# CONFIDENCE SHIFT
# ============================================================

confidence_rows = []

# We already need the confidence for each condition.
# Recompute condition confidence arrays once.

condition_confidences = {}

for condition in conditions:

    texts = shift_df[
        condition
    ].fillna("").astype(str).tolist()

    embeddings = get_embeddings(texts)

    probabilities = classifier.predict_proba(
        embeddings
    )

    condition_confidences[condition] = (
        probabilities.max(axis=1)
    )

for condition in conditions:

    mean_conf = condition_confidences[
        condition
    ].mean()

    if condition == "original":

        shift_from_original = 0.0

    else:

        shift_from_original = (
            condition_confidences[condition]
            - condition_confidences["original"]
        ).mean()

    confidence_rows.append({
        "condition": condition,
        "mean_confidence": mean_conf,
        "mean_confidence_shift_from_original":
            shift_from_original
    })

confidence_df = pd.DataFrame(
    confidence_rows
)

confidence_df.to_csv(
    CONFIDENCE_OUTPUT,
    index=False
)

print(
    f"\nSaved confidence shifts to:\n{CONFIDENCE_OUTPUT}"
)

# ============================================================
# SAVE ALL PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    all_predictions
)

predictions_df.to_csv(
    PREDICTION_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nSaved predictions to:\n{PREDICTION_OUTPUT}"
)

# ============================================================
# TRAINING INFORMATION
# ============================================================

training_info = pd.DataFrame([
    {
        "model": MODEL_NAME,
        "approach": "Frozen XLM-R embeddings + Logistic Regression",
        "training_samples": len(train_df),
        "embedding_dimension":
            int(X_train_embeddings.shape[1]),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "random_seed": SEED,
        "num_conditions": len(conditions)
    }
])

training_info.to_csv(
    TRAINING_INFO_OUTPUT,
    index=False
)

# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "model_name": MODEL_NAME,
    "approach": "frozen_embeddings_logistic_regression",
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "random_seed": SEED,
    "device": str(DEVICE),
    "train_file": TRAIN_FILE,
    "script_shift_file": SCRIPT_SHIFT_FILE,
    "conditions": conditions
}

with open(
    CONFIG_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("XLM-R EXPERIMENT COMPLETE")
print("=" * 80)

print("\nFiles created:")

print(f"  {RESULTS_OUTPUT}")
print(f"  {FLIP_OUTPUT}")
print(f"  {CONFIDENCE_OUTPUT}")
print(f"  {TRAINING_INFO_OUTPUT}")
print(f"  {CONFIG_OUTPUT}")
print(f"  {PREDICTION_OUTPUT}")

print("\n" + results_df[
    [
        "Condition",
        "Accuracy",
        "Balanced Accuracy",
        "Macro-F1",
        "Mean Confidence",
        "Prediction Flip Rate"
    ]
].to_string(index=False))

print("\nDone.")