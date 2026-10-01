import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/script_shift/validation_script_shift_full.csv"

OUTPUT_FILE = "results/predictions/semantic_audit_automated.csv"
SUMMARY_FILE = "results/tables/semantic_audit_summary.csv"

SAMPLE_SIZE = 200

# Similarity threshold for automated screening.
# This is NOT a human annotation.
SIMILARITY_THRESHOLD = 0.70

RANDOM_STATE = 42

CONDITIONS = [
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise"
]

# Multilingual sentence embedding model
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AUTOMATED SEMANTIC-PRESERVATION SCREENING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

required_columns = ["id", "original_text"] + CONDITIONS

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Select 200 examples
if len(df) > SAMPLE_SIZE:
    audit_df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).copy()
else:
    audit_df = df.copy()

audit_df = audit_df.reset_index(drop=True)

print("Audit examples:", len(audit_df))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading multilingual sentence embedding model...")
print(MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nEncoding original sentences...")

original_texts = (
    audit_df["original_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

original_embeddings = model.encode(
    original_texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


# ============================================================
# COMPARE EACH TRANSFORMATION
# ============================================================

records = []

for condition in CONDITIONS:

    print("\n" + "-" * 70)
    print("Condition:", condition)
    print("-" * 70)

    transformed_texts = (
        audit_df[condition]
        .fillna("")
        .astype(str)
        .tolist()
    )

    transformed_embeddings = model.encode(
        transformed_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = np.sum(
        original_embeddings * transformed_embeddings,
        axis=1
    )

    for i in range(len(audit_df)):

        similarity = float(similarities[i])

        if similarity >= SIMILARITY_THRESHOLD:
            automated_label = 1
        else:
            automated_label = 0

        records.append({
            "id": audit_df.loc[i, "id"],
            "condition": condition,
            "original_text": audit_df.loc[i, "original_text"],
            "transformed_text": audit_df.loc[i, condition],
            "semantic_similarity": similarity,
            "automated_preserved": automated_label
        })

    print(
        "Mean similarity:",
        round(float(np.mean(similarities)), 4)
    )

    print(
        "Median similarity:",
        round(float(np.median(similarities)), 4)
    )

    print(
        "Automatically classified as preserved:",
        int(np.sum(similarities >= SIMILARITY_THRESHOLD)),
        "/",
        len(similarities)
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results = pd.DataFrame(records)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(SUMMARY_FILE),
    exist_ok=True
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

summary_records = []

for condition in CONDITIONS:

    subset = results[
        results["condition"] == condition
    ]

    preserved = (
        subset["automated_preserved"] == 1
    ).sum()

    total = len(subset)

    summary_records.append({
        "condition": condition,
        "n_examples": total,
        "mean_similarity": round(
            subset["semantic_similarity"].mean(),
            4
        ),
        "median_similarity": round(
            subset["semantic_similarity"].median(),
            4
        ),
        "min_similarity": round(
            subset["semantic_similarity"].min(),
            4
        ),
        "max_similarity": round(
            subset["semantic_similarity"].max(),
            4
        ),
        "automated_preserved": int(preserved),
        "automated_changed": int(total - preserved),
        "automated_preservation_rate": round(
            preserved / total,
            4
        )
    })

summary = pd.DataFrame(summary_records)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("AUTOMATED AUDIT COMPLETE")
print("=" * 70)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nDetailed results saved to:")
print(OUTPUT_FILE)

print("\nSummary saved to:")
print(SUMMARY_FILE)

print("\nNOTE:")
print(
    "This is an automated embedding-based semantic screening, "
    "not a human annotation study."
)

print("=" * 70)