"""
SCRIPT-CERT
MuRIL Baseline + SCRIPT-SHIFT Evaluation

Model:
    google/muril-base-cased

Training:
    data/processed/train_clean.csv
    4000 labeled training samples

Evaluation:
    data/script_shift/script_shift_base.csv
    800 validation samples
    6 SCRIPT-SHIFT conditions

Conditions:
    1. original
    2. romanized
    3. telugu_script
    4. mixed_script
    5. mild_noise
    6. severe_noise

MuRIL is frozen.
A Logistic Regression classifier is trained on MuRIL embeddings.

Outputs:
    results/tables/muril_script_shift_results.csv
    results/predictions/muril_script_shift_predictions.csv
    results/tables/muril_training_info.csv
    results/tables/muril_config.json
    results/embeddings/muril_train_embeddings.npy
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import random
import time
import warnings

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_NAME = "google/muril-base-cased"

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_clean.csv"
)

SCRIPT_SHIFT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "script_shift",
    "script_shift_base.csv"
)

RESULTS_TABLES = os.path.join(
    PROJECT_ROOT,
    "results",
    "tables"
)

RESULTS_PREDICTIONS = os.path.join(
    PROJECT_ROOT,
    "results",
    "predictions"
)

RESULTS_EMBEDDINGS = os.path.join(
    PROJECT_ROOT,
    "results",
    "embeddings"
)


# Create output directories
os.makedirs(RESULTS_TABLES, exist_ok=True)
os.makedirs(RESULTS_PREDICTIONS, exist_ok=True)
os.makedirs(RESULTS_EMBEDDINGS, exist_ok=True)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. HARDWARE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)

    print("=" * 70)
    print("GPU AVAILABLE")
    print("=" * 70)
    print(f"Device: {DEVICE_NAME}")

else:
    DEVICE = torch.device("cpu")

    print("=" * 70)
    print("GPU NOT AVAILABLE")
    print("=" * 70)
    print("Using CPU.")


# ============================================================
# 5. HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 16 if DEVICE.type == "cpu" else 64

MAX_LENGTH = 64

CLASSIFIER_MAX_ITER = 2000

print()
print("MuRIL configuration:")
print(f"Model       : {MODEL_NAME}")
print(f"Device      : {DEVICE}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Max length  : {MAX_LENGTH}")
print(f"Random seed : {SEED}")


# ============================================================
# 6. LOAD TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

train_df = pd.read_csv(TRAIN_FILE)

print(f"Training file: {TRAIN_FILE}")
print(f"Training shape: {train_df.shape}")
print(f"Training columns: {train_df.columns.tolist()}")


# ------------------------------------------------------------
# Normalize training column names
# ------------------------------------------------------------

if "text" in train_df.columns and "label" in train_df.columns:

    train_text_col = "text"
    train_label_col = "label"

elif "Comments" in train_df.columns and "Label" in train_df.columns:

    train_text_col = "Comments"
    train_label_col = "Label"

else:

    raise ValueError(
        "Training dataset must contain either:\n"
        "    text + label\n"
        "or:\n"
        "    Comments + Label"
    )


train_df = train_df[
    [train_text_col, train_label_col]
].copy()

train_df.columns = ["text", "label"]

train_df["text"] = train_df["text"].astype(str).str.strip()
train_df["label"] = train_df["label"].astype(str).str.strip().str.lower()

train_df = train_df[
    (train_df["text"] != "") &
    (train_df["text"].notna()) &
    (train_df["label"].notna())
].reset_index(drop=True)


print(f"Valid training rows: {len(train_df)}")
print()
print("Training label distribution:")
print(train_df["label"].value_counts())


# ============================================================
# 7. LOAD SCRIPT-SHIFT DATA
# ============================================================

print()
print("=" * 70)
print("LOADING SCRIPT-SHIFT DATA")
print("=" * 70)

if not os.path.exists(SCRIPT_SHIFT_FILE):
    raise FileNotFoundError(
        f"SCRIPT-SHIFT file not found:\n{SCRIPT_SHIFT_FILE}"
    )

shift_df = pd.read_csv(SCRIPT_SHIFT_FILE)

print(f"SCRIPT-SHIFT file: {SCRIPT_SHIFT_FILE}")
print(f"SCRIPT-SHIFT shape: {shift_df.shape}")
print(f"Columns: {shift_df.columns.tolist()}")


# ============================================================
# 8. REQUIRED SCRIPT-SHIFT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "id",
    "original_text",
    "label",
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise",
]

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in shift_df.columns
]

if missing_columns:

    raise ValueError(
        "SCRIPT-SHIFT dataset is missing columns:\n"
        + "\n".join(missing_columns)
    )


# Keep exactly the required columns
shift_df = shift_df[REQUIRED_COLUMNS].copy()


# ============================================================
# 9. CLEAN SCRIPT-SHIFT DATA
# ============================================================

shift_df["label"] = (
    shift_df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

CONDITIONS = [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise",
]


for condition in CONDITIONS:

    shift_df[condition] = (
        shift_df[condition]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# Remove rows where original text is empty
shift_df = shift_df[
    shift_df["original"].str.len() > 0
].reset_index(drop=True)


print()
print(f"Valid SCRIPT-SHIFT rows: {len(shift_df)}")

if len(shift_df) != 800:

    print()
    print(
        "WARNING: Expected 800 SCRIPT-SHIFT validation rows, "
        f"but found {len(shift_df)}."
    )

    print(
        "The script will continue only if all transformation "
        "columns contain valid text."
    )


# ============================================================
# 10. VALIDATE TRANSFORMATION COLUMNS
# ============================================================

print()
print("=" * 70)
print("SCRIPT-SHIFT QUALITY CHECK")
print("=" * 70)

quality_rows = []

for condition in CONDITIONS:

    non_empty = (
        shift_df[condition]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    unique_count = shift_df[condition].nunique()

    quality_rows.append(
        {
            "condition": condition,
            "rows": len(shift_df),
            "non_empty": int(non_empty),
            "unique": int(unique_count),
        }
    )

quality_df = pd.DataFrame(quality_rows)

print(quality_df.to_string(index=False))


# ------------------------------------------------------------
# Stop if transformation data is empty
# ------------------------------------------------------------

invalid_conditions = quality_df[
    quality_df["non_empty"] != len(shift_df)
]["condition"].tolist()


if invalid_conditions:

    raise ValueError(
        "\nThese SCRIPT-SHIFT columns contain empty values:\n"
        + str(invalid_conditions)
        + "\n\nRun:\n"
        "python src/transformations/fix_script_shift_data.py\n"
        "before running MuRIL."
    )


# ============================================================
# 11. CHECK LABELS
# ============================================================

print()
print("SCRIPT-SHIFT labels:")
print(shift_df["label"].value_counts())


train_labels = sorted(train_df["label"].unique().tolist())
shift_labels = sorted(shift_df["label"].unique().tolist())

print()
print(f"Training labels: {train_labels}")
print(f"SHIFT labels:    {shift_labels}")


if set(train_labels) != set(shift_labels):

    raise ValueError(
        "Training and SCRIPT-SHIFT label sets do not match.\n"
        f"Training: {train_labels}\n"
        f"SHIFT: {shift_labels}"
    )


# ============================================================
# 12. LOAD MuRIL TOKENIZER
# ============================================================

print()
print("=" * 70)
print("LOADING MuRIL TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 13. LOAD MuRIL MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MuRIL MODEL")
print("=" * 70)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)

model.eval()

# Freeze MuRIL
for parameter in model.parameters():
    parameter.requires_grad = False

print("MuRIL loaded.")
print("MuRIL parameters frozen.")
print(f"Device: {DEVICE}")


# ============================================================
# 14. MEAN POOLING FUNCTION
# ============================================================

def mean_pooling(
    model_output,
    attention_mask
):
    """
    Mean-pool token embeddings while ignoring padding tokens.
    """

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    sum_embeddings = torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1
    )

    sum_mask = torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9
    )

    return sum_embeddings / sum_mask


# ============================================================
# 15. EMBEDDING EXTRACTION FUNCTION
# ============================================================

@torch.no_grad()
def extract_embeddings(
    texts,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
):
    """
    Convert text into frozen MuRIL embeddings.
    """

    embeddings = []

    total = len(texts)

    start_time = time.time()

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total
        )

        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        pooled = mean_pooling(
            outputs,
            encoded["attention_mask"]
        )

        pooled = pooled.cpu().numpy()

        embeddings.append(pooled)

        processed = end
        percent = processed / total * 100

        elapsed = time.time() - start_time

        print(
            f"\rEmbedding {processed}/{total} "
            f"({percent:.1f}%) | "
            f"Elapsed: {elapsed/60:.1f} min",
            end=""
        )

    print()

    return np.vstack(embeddings)


# ============================================================
# 16. TRAIN / VALIDATION SPLIT
# ============================================================

print()
print("=" * 70)
print("CREATING TRAINING SPLIT")
print("=" * 70)

from sklearn.model_selection import train_test_split

train_texts = train_df["text"].values
train_labels_array = train_df["label"].values

(
    X_train_text,
    X_unused_text,
    y_train,
    y_unused
) = train_test_split(
    train_texts,
    train_labels_array,
    test_size=0.20,
    random_state=SEED,
    stratify=train_labels_array
)

print(f"Training samples used for MuRIL: {len(X_train_text)}")
print(f"Holdout samples not used:         {len(X_unused_text)}")

print()
print("MuRIL training label distribution:")
print(pd.Series(y_train).value_counts())


# ============================================================
# 17. EXTRACT TRAINING EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("EXTRACTING MuRIL TRAINING EMBEDDINGS")
print("=" * 70)

train_embedding_file = os.path.join(
    RESULTS_EMBEDDINGS,
    "muril_train_embeddings.npy"
)

train_embeddings = None

if os.path.exists(train_embedding_file):

    print(
        "Existing training embeddings found."
    )

    try:

        cached_embeddings = np.load(
            train_embedding_file
        )

        if cached_embeddings.shape[0] == len(X_train_text):

            train_embeddings = cached_embeddings

            print(
                f"Loaded cached embeddings: "
                f"{train_embeddings.shape}"
            )

        else:

            print(
                "Cached embedding count does not match."
            )

    except Exception as e:

        print(
            f"Could not load cached embeddings: {e}"
        )


if train_embeddings is None:

    train_embeddings = extract_embeddings(
        X_train_text
    )

    np.save(
        train_embedding_file,
        train_embeddings
    )

    print(
        f"Saved training embeddings: "
        f"{train_embedding_file}"
    )


print()
print(
    f"Training embedding shape: "
    f"{train_embeddings.shape}"
)


# ============================================================
# 18. TRAIN LOGISTIC REGRESSION CLASSIFIER
# ============================================================

print()
print("=" * 70)
print("TRAINING MuRIL + LOGISTIC REGRESSION")
print("=" * 70)

classifier = LogisticRegression(
    max_iter=CLASSIFIER_MAX_ITER,
    class_weight="balanced",
    random_state=SEED,
    solver="lbfgs"
)

classifier.fit(
    train_embeddings,
    y_train
)

print("Classifier training completed.")

print()
print(
    f"Classifier classes: "
    f"{classifier.classes_.tolist()}"
)


# ============================================================
# 19. SAVE TRAINING INFORMATION
# ============================================================

training_info = pd.DataFrame(
    [
        {
            "model": MODEL_NAME,
            "embedding_method": "Mean Pooling",
            "classifier": "LogisticRegression",
            "training_samples": len(X_train_text),
            "embedding_dimension": train_embeddings.shape[1],
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "device": str(DEVICE),
            "seed": SEED,
            "class_weight": "balanced",
            "max_iter": CLASSIFIER_MAX_ITER,
        }
    ]
)

training_info_file = os.path.join(
    RESULTS_TABLES,
    "muril_training_info.csv"
)

training_info.to_csv(
    training_info_file,
    index=False
)


# ============================================================
# 20. EXTRACT ALL SCRIPT-SHIFT TEXT
# ============================================================

print()
print("=" * 70)
print("EXTRACTING SCRIPT-SHIFT EMBEDDINGS")
print("=" * 70)

condition_embeddings = {}

for condition in CONDITIONS:

    print()
    print("-" * 70)
    print(f"Condition: {condition}")
    print("-" * 70)

    texts = shift_df[condition].tolist()

    embeddings = extract_embeddings(
        texts
    )

    condition_embeddings[condition] = embeddings

    print(
        f"{condition} embedding shape: "
        f"{embeddings.shape}"
    )


# ============================================================
# 21. EVALUATION FUNCTION
# ============================================================

def evaluate_condition(
    condition,
    embeddings,
    true_labels
):
    """
    Evaluate one SCRIPT-SHIFT condition.
    """

    probabilities = classifier.predict_proba(
        embeddings
    )

    predictions = classifier.predict(
        embeddings
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro"
    )

    macro_precision = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    result = {
        "Condition": condition,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro F1": macro_f1,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Mean Confidence": confidence.mean(),
    }

    return (
        result,
        predictions,
        probabilities,
        confidence
    )


# ============================================================
# 22. RUN ALL CONDITIONS
# ============================================================

print()
print("=" * 70)
print("RUNNING SCRIPT-SHIFT EVALUATION")
print("=" * 70)

true_labels = shift_df["label"].values

results = []

prediction_records = []

all_predictions = {}

all_probabilities = {}

all_confidences = {}


for condition in CONDITIONS:

    print()
    print(f"Evaluating: {condition}")

    (
        result,
        predictions,
        probabilities,
        confidence
    ) = evaluate_condition(
        condition,
        condition_embeddings[condition],
        true_labels
    )

    results.append(result)

    all_predictions[condition] = predictions

    all_probabilities[condition] = probabilities

    all_confidences[condition] = confidence

    # --------------------------------------------------------
    # Prediction records
    # --------------------------------------------------------

    for i in range(len(shift_df)):

        prediction_records.append(
            {
                "id": shift_df.iloc[i]["id"],
                "true_label": true_labels[i],
                "condition": condition,
                "text": shift_df.iloc[i][condition],
                "prediction": predictions[i],
                "confidence": confidence[i],
                "probability_hate": probabilities[i][
                    list(classifier.classes_).index("hate")
                ],
                "probability_non_hate": probabilities[i][
                    list(classifier.classes_).index("non-hate")
                ],
            }
        )

    # --------------------------------------------------------
    # Print classification report
    # --------------------------------------------------------

    print()

    print(
        classification_report(
            true_labels,
            predictions,
            labels=classifier.classes_,
            zero_division=0
        )
    )


# ============================================================
# 23. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

print()
print("=" * 70)
print("FINAL MuRIL SCRIPT-SHIFT RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 24. SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_TABLES,
    "muril_script_shift_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print()
print(
    f"Saved results:\n{results_file}"
)


# ============================================================
# 25. SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    prediction_records
)

predictions_file = os.path.join(
    RESULTS_PREDICTIONS,
    "muril_script_shift_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)

print(
    f"Saved predictions:\n{predictions_file}"
)


# ============================================================
# 26. CALCULATE PREDICTION FLIP RATES
# ============================================================

print()
print("=" * 70)
print("PREDICTION FLIP ANALYSIS")
print("=" * 70)

original_predictions = all_predictions["original"]

flip_results = []

for condition in CONDITIONS:

    condition_predictions = all_predictions[
        condition
    ]

    flip_count = np.sum(
        condition_predictions != original_predictions
    )

    flip_rate = (
        flip_count /
        len(original_predictions)
    )

    flip_results.append(
        {
            "Condition": condition,
            "Flip Count": int(flip_count),
            "Flip Rate": flip_rate,
        }
    )

    print(
        f"{condition:20s} "
        f"Flips: {flip_count:4d} "
        f"Rate: {flip_rate:.4f}"
    )


flip_df = pd.DataFrame(
    flip_results
)

flip_file = os.path.join(
    RESULTS_TABLES,
    "muril_prediction_flip_rates.csv"
)

flip_df.to_csv(
    flip_file,
    index=False
)


# ============================================================
# 27. CONFIDENCE SHIFT
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE SHIFT")
print("=" * 70)

original_confidence = all_confidences[
    "original"
]

confidence_results = []

for condition in CONDITIONS:

    condition_confidence = all_confidences[
        condition
    ]

    mean_change = (
        condition_confidence -
        original_confidence
    ).mean()

    mean_absolute_change = np.abs(
        condition_confidence -
        original_confidence
    ).mean()

    confidence_results.append(
        {
            "Condition": condition,
            "Mean Confidence": condition_confidence.mean(),
            "Mean Confidence Change": mean_change,
            "Mean Absolute Confidence Change":
                mean_absolute_change,
        }
    )

    print(
        f"{condition:20s} "
        f"Mean: {condition_confidence.mean():.4f} "
        f"Change: {mean_change:.4f}"
    )


confidence_df = pd.DataFrame(
    confidence_results
)

confidence_file = os.path.join(
    RESULTS_TABLES,
    "muril_confidence_shift.csv"
)

confidence_df.to_csv(
    confidence_file,
    index=False
)


# ============================================================
# 28. SAVE CONFIGURATION
# ============================================================

config = {
    "model": MODEL_NAME,
    "embedding_method": "mean_pooling",
    "classifier": "LogisticRegression",
    "classifier_parameters": {
        "max_iter": CLASSIFIER_MAX_ITER,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": SEED
    },
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "seed": SEED,
    "device": str(DEVICE),
    "training_file": TRAIN_FILE,
    "script_shift_file": SCRIPT_SHIFT_FILE,
    "training_samples": int(len(X_train_text)),
    "script_shift_samples": int(len(shift_df)),
    "conditions": CONDITIONS,
}


config_file = os.path.join(
    RESULTS_TABLES,
    "muril_config.json"
)

with open(
    config_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# 29. SAVE EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("SAVING SCRIPT-SHIFT EMBEDDINGS")
print("=" * 70)

for condition in CONDITIONS:

    output_file = os.path.join(
        RESULTS_EMBEDDINGS,
        f"muril_{condition}_embeddings.npy"
    )

    np.save(
        output_file,
        condition_embeddings[condition]
    )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 80)
print("MuRIL SCRIPT-SHIFT EXPERIMENT COMPLETED")
print("=" * 80)

print()
print("MODEL")
print("-----")
print(f"MuRIL: {MODEL_NAME}")
print("MuRIL frozen: YES")
print("Classifier: Logistic Regression")

print()
print("DATA")
print("----")
print(f"Training samples used : {len(X_train_text)}")
print(f"SCRIPT-SHIFT samples  : {len(shift_df)}")

print()
print("CONDITIONS")
print("----------")

for condition in CONDITIONS:
    print(f"  ✓ {condition}")

print()
print("RESULTS")
print("-------")

print(
    results_df[
        [
            "Condition",
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1",
            "Mean Confidence"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("OUTPUT FILES")
print("------------")

print(results_file)
print(predictions_file)
print(training_info_file)
print(config_file)
print(flip_file)
print(confidence_file)

print()
print("=" * 80)
print("DONE")
print("=" * 80)