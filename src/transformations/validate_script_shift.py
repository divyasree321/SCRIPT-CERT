from pathlib import Path
import pandas as pd
import random
import re


# ============================================================
# CONFIG
# ============================================================

SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "script_shift"
    / "validation_script_shift_full.csv"
)


# ============================================================
# SCRIPT DETECTION
# ============================================================

def contains_telugu(text):
    text = str(text)

    return any(
        "\u0C00" <= ch <= "\u0C7F"
        for ch in text
    )


def contains_latin(text):
    text = str(text)

    return any(
        ("A" <= ch <= "Z") or
        ("a" <= ch <= "z")
        for ch in text
    )


def script_type(text):

    has_telugu = contains_telugu(text)
    has_latin = contains_latin(text)

    if has_telugu and has_latin:
        return "Mixed"

    elif has_telugu:
        return "Telugu"

    elif has_latin:
        return "Latin/Romanized"

    else:
        return "Other"


# ============================================================
# TEXT STATISTICS
# ============================================================

def text_stats(text):

    text = str(text)

    return {
        "characters": len(text),
        "words": len(text.split()),
        "telugu_chars": sum(
            "\u0C00" <= ch <= "\u0C7F"
            for ch in text
        ),
        "latin_chars": sum(
            ("A" <= ch <= "Z") or
            ("a" <= ch <= "z")
            for ch in text
        )
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("SCRIPT-CERT TRANSFORMATION VALIDATION")
print("=" * 80)

print("\nLoading:")

print(INPUT_FILE)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("\nDataset shape:", df.shape)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

conditions = [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise"
]

required = [
    "original_text",
    "label"
] + conditions

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# CONDITION SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("CONDITION SUMMARY")
print("=" * 80)

for condition in conditions:

    changed = (
        df[condition].astype(str)
        != df["original_text"].astype(str)
    )

    empty = (
        df[condition].fillna("").astype(str).str.strip()
        == ""
    )

    print(f"\n{condition}")
    print("-" * 50)

    print(
        "Changed:",
        changed.sum(),
        f"({changed.mean() * 100:.2f}%)"
    )

    print(
        "Unchanged:",
        (~changed).sum()
    )

    print(
        "Empty:",
        empty.sum()
    )


# ============================================================
# SCRIPT DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("SCRIPT DISTRIBUTION")
print("=" * 80)

for condition in conditions:

    print(f"\n{condition}:")

    distribution = (
        df[condition]
        .apply(script_type)
        .value_counts()
    )

    print(distribution)


# ============================================================
# RANDOM EXAMPLES
# ============================================================

print("\n")
print("=" * 80)
print("RANDOM TRANSFORMATION EXAMPLES")
print("=" * 80)

sample_size = min(
    20,
    len(df)
)

sample_indices = random.sample(
    list(df.index),
    sample_size
)

for number, idx in enumerate(
    sample_indices,
    start=1
):

    row = df.loc[idx]

    print("\n")
    print("-" * 80)
    print(f"EXAMPLE {number}")
    print("-" * 80)

    print("\nTRUE LABEL:")
    print(row["label"])

    print("\nORIGINAL:")
    print(row["original_text"])

    print("\nROMANIZED:")
    print(row["romanized"])

    print("\nTELUGU SCRIPT:")
    print(row["telugu_script"])

    print("\nMIXED SCRIPT:")
    print(row["mixed_script"])

    print("\nMILD NOISE:")
    print(row["mild_noise"])

    print("\nSEVERE NOISE:")
    print(row["severe_noise"])


# ============================================================
# LENGTH ANALYSIS
# ============================================================

print("\n")
print("=" * 80)
print("AVERAGE TEXT LENGTH")
print("=" * 80)

for condition in conditions:

    lengths = (
        df[condition]
        .astype(str)
        .str.len()
    )

    words = (
        df[condition]
        .astype(str)
        .str.split()
        .str.len()
    )

    print(
        f"{condition:20s} "
        f"characters={lengths.mean():.1f} "
        f"words={words.mean():.1f}"
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\n")
print("=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)

for condition in conditions:

    duplicates = (
        df[condition]
        .astype(str)
        .duplicated()
        .sum()
    )

    print(
        f"{condition:20s}: "
        f"{duplicates} duplicates"
    )


print("\n")
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)