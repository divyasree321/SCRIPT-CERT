import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_FILE = "results/predictions/xlmr_script_shift_predictions.csv"

OUTPUT_FILE = "results/tables/xlmr_conformal_prediction_final.csv"
CONFIG_FILE = "results/tables/xlmr_conformal_config.json"

RANDOM_STATE = 42
TARGET_COVERAGE = 0.90

CONDITIONS = [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise"
]


# ============================================================
# CONFORMAL QUANTILE
# ============================================================

def conformal_quantile(scores, target_coverage):
    """
    Split-conformal quantile with finite-sample correction.

    Nonconformity score:
        1 - probability assigned to the true class
    """

    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    alpha = 1.0 - target_coverage

    quantile_level = np.ceil((n + 1) * (1.0 - alpha)) / n

    quantile_level = min(quantile_level, 1.0)

    try:
        q = np.quantile(
            scores,
            quantile_level,
            method="higher"
        )
    except TypeError:
        q = np.quantile(
            scores,
            quantile_level,
            interpolation="higher"
        )

    return float(q)


# ============================================================
# CREATE PREDICTION SETS
# ============================================================

def create_prediction_sets(probabilities, threshold):
    """
    Create conformal prediction sets.

    A class is included when:

        1 - P(class) <= threshold

    Equivalent to:

        P(class) >= 1 - threshold
    """

    prediction_sets = []

    for probs in probabilities:

        selected = []

        for class_index, probability in enumerate(probs):

            if (1.0 - probability) <= threshold:
                selected.append(class_index)

        # Safety fallback:
        # never allow an empty prediction set.
        if len(selected) == 0:
            selected = [int(np.argmax(probs))]

        prediction_sets.append(selected)

    return prediction_sets


# ============================================================
# EVALUATE PREDICTION SETS
# ============================================================

def evaluate_prediction_sets(prediction_sets, y_true):

    covered = []

    set_sizes = []

    singleton_count = 0
    full_set_count = 0
    empty_count = 0

    for pred_set, true_label in zip(prediction_sets, y_true):

        covered.append(int(true_label in pred_set))

        set_sizes.append(len(pred_set))

        if len(pred_set) == 1:
            singleton_count += 1

        elif len(pred_set) == 2:
            full_set_count += 1

        elif len(pred_set) == 0:
            empty_count += 1

    coverage = np.mean(covered)

    average_set_size = np.mean(set_sizes)

    review_rate = np.mean(
        [size > 1 for size in set_sizes]
    )

    return {
        "Coverage": float(coverage),
        "Average Prediction Set Size": float(average_set_size),
        "Singleton Sets": int(singleton_count),
        "Full Sets": int(full_set_count),
        "Empty Sets": int(empty_count),
        "Human Review Rate": float(review_rate),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("XLM-R CONFORMAL PREDICTION")
print("=" * 80)

print("\nLoading predictions...")

df = pd.read_csv(PREDICTION_FILE)

print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "id",
    "true_label",
    "condition",
    "prediction",
    "probability_hate",
    "probability_non_hate"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_TO_INDEX = {
    "hate": 0,
    "non-hate": 1
}

print("\nClass mapping:")
print(CLASS_TO_INDEX)


# ============================================================
# ORIGINAL DATA
# ============================================================

original = df[
    df["condition"] == "original"
].copy()

original = original.sort_values("id").reset_index(drop=True)

print("\nOriginal condition:")
print(original.shape)


# ============================================================
# CHECK 800 SAMPLES
# ============================================================

if len(original) != 800:
    raise ValueError(
        f"Expected 800 original samples, found {len(original)}"
    )


# ============================================================
# TRUE LABELS
# ============================================================

y_original = np.array([
    CLASS_TO_INDEX[str(label)]
    for label in original["true_label"]
])


# ============================================================
# PROBABILITIES
# ============================================================

original_probabilities = original[
    ["probability_hate", "probability_non_hate"]
].values.astype(float)


# ============================================================
# CALIBRATION / FINAL EVALUATION SPLIT
# ============================================================

indices = np.arange(len(original))

calibration_indices, evaluation_indices = train_test_split(
    indices,
    test_size=400,
    random_state=RANDOM_STATE,
    stratify=y_original
)

print("\nSplit:")
print(f"Calibration samples: {len(calibration_indices)}")
print(f"Final evaluation samples: {len(evaluation_indices)}")


# ============================================================
# CALIBRATION NONCONFORMITY SCORES
# ============================================================

calibration_probabilities = original_probabilities[
    calibration_indices
]

calibration_labels = y_original[
    calibration_indices
]


true_probabilities = np.array([
    calibration_probabilities[i, calibration_labels[i]]
    for i in range(len(calibration_labels))
])


nonconformity_scores = 1.0 - true_probabilities


# ============================================================
# CALCULATE CONFORMAL THRESHOLD
# ============================================================

threshold = conformal_quantile(
    nonconformity_scores,
    TARGET_COVERAGE
)

probability_threshold = 1.0 - threshold


print("\n" + "=" * 80)
print("CONFORMAL CALIBRATION")
print("=" * 80)

print(f"Target coverage: {TARGET_COVERAGE:.2%}")
print(f"Conformal threshold: {threshold:.6f}")
print(f"Probability threshold: {probability_threshold:.6f}")


# ============================================================
# EVALUATE ALL CONDITIONS
# ============================================================

results = []

for condition in CONDITIONS:

    print("\n" + "-" * 70)
    print(f"Condition: {condition}")
    print("-" * 70)

    condition_df = df[
        df["condition"] == condition
    ].copy()

    condition_df = condition_df.sort_values(
        "id"
    ).reset_index(drop=True)

    if len(condition_df) != 800:
        raise ValueError(
            f"{condition}: expected 800 rows, "
            f"found {len(condition_df)}"
        )

    probabilities = condition_df[
        ["probability_hate", "probability_non_hate"]
    ].values.astype(float)

    labels = np.array([
        CLASS_TO_INDEX[str(label)]
        for label in condition_df["true_label"]
    ])

    # Use the same final evaluation indices
    # across all script-shift conditions.
    eval_probabilities = probabilities[
        evaluation_indices
    ]

    eval_labels = labels[
        evaluation_indices
    ]

    # Create prediction sets
    prediction_sets = create_prediction_sets(
        eval_probabilities,
        threshold
    )

    metrics = evaluate_prediction_sets(
        prediction_sets,
        eval_labels
    )

    metrics["Condition"] = condition
    metrics["Target Coverage"] = TARGET_COVERAGE
    metrics["Conformal Threshold"] = threshold
    metrics["Probability Threshold"] = probability_threshold
    metrics["Calibration Samples"] = len(calibration_indices)
    metrics["Final Evaluation Samples"] = len(evaluation_indices)

    results.append(metrics)

    print(
        f"Coverage: "
        f"{metrics['Coverage']:.4f}"
    )

    print(
        f"Average Set Size: "
        f"{metrics['Average Prediction Set Size']:.4f}"
    )

    print(
        f"Human Review Rate: "
        f"{metrics['Human Review Rate']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

column_order = [
    "Condition",
    "Target Coverage",
    "Coverage",
    "Average Prediction Set Size",
    "Singleton Sets",
    "Full Sets",
    "Empty Sets",
    "Human Review Rate",
    "Conformal Threshold",
    "Probability Threshold",
    "Calibration Samples",
    "Final Evaluation Samples"
]

results_df = results_df[column_order]


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "method": "Split Conformal Prediction",
    "model": "XLM-R",
    "target_coverage": TARGET_COVERAGE,
    "random_state": RANDOM_STATE,
    "calibration_samples": len(calibration_indices),
    "final_evaluation_samples": len(evaluation_indices),
    "calibration_condition": "original",
    "conditions_evaluated": CONDITIONS,
    "nonconformity_score": "1 - probability_of_true_class",
    "prediction_rule": "1 - probability <= conformal_threshold"
}

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("XLM-R CONFORMAL PREDICTION COMPLETE")
print("=" * 80)

print("\nFinal Results:")
print(
    results_df.to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_FILE)

print(CONFIG_FILE)

print("\nDONE.")