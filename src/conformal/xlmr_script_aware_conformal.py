import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_FILE = (
    "results/predictions/xlmr_script_shift_predictions.csv"
)

BASE_FILE = (
    "data/script_shift/validation_script_shift_full.csv"
)

OUTPUT_FILE = (
    "results/tables/xlmr_script_aware_conformal_final.csv"
)

CONFIG_FILE = (
    "results/tables/xlmr_script_aware_conformal_config.json"
)

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

# Very small groups cannot reliably support their own
# conformal calibration. They will use the global threshold.
MIN_GROUP_CALIBRATION = 30


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_TO_INDEX = {
    "hate": 0,
    "non-hate": 1
}


# ============================================================
# CONFORMAL QUANTILE
# ============================================================

def conformal_quantile(scores, target_coverage):

    scores = np.asarray(scores, dtype=float)

    n = len(scores)

    alpha = 1.0 - target_coverage

    quantile_level = (
        np.ceil((n + 1) * (1.0 - alpha)) / n
    )

    quantile_level = min(
        quantile_level,
        1.0
    )

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
# CREATE PREDICTION SET
# ============================================================

def create_prediction_set(
    probabilities,
    threshold
):

    selected = []

    for class_index, probability in enumerate(
        probabilities
    ):

        if (
            1.0 - probability
        ) <= threshold:

            selected.append(class_index)

    # Safety fallback
    if len(selected) == 0:

        selected = [
            int(np.argmax(probabilities))
        ]

    return selected


# ============================================================
# EVALUATE
# ============================================================

def evaluate_prediction_sets(
    prediction_sets,
    labels
):

    covered = []

    set_sizes = []

    singleton = 0
    full = 0
    empty = 0

    for pred_set, true_label in zip(
        prediction_sets,
        labels
    ):

        covered.append(
            int(true_label in pred_set)
        )

        set_sizes.append(
            len(pred_set)
        )

        if len(pred_set) == 1:

            singleton += 1

        elif len(pred_set) == 2:

            full += 1

        elif len(pred_set) == 0:

            empty += 1

    return {
        "Coverage": np.mean(covered),

        "Average Prediction Set Size":
            np.mean(set_sizes),

        "Singleton Sets":
            singleton,

        "Full Sets":
            full,

        "Empty Sets":
            empty,

        "Human Review Rate":
            np.mean(
                np.array(set_sizes) > 1
            )
    }


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 80)
print("XLM-R SCRIPT-AWARE CONFORMAL PREDICTION")
print("=" * 80)

print("\nLoading XLM-R predictions...")

pred_df = pd.read_csv(
    PREDICTION_FILE
)

print(
    "Prediction shape:",
    pred_df.shape
)

print("\nLoading script information...")

base_df = pd.read_csv(
    BASE_FILE,
    encoding="utf-8-sig"
)

print(
    "Base shape:",
    base_df.shape
)


# ============================================================
# CHECK COLUMNS
# ============================================================

print("\nBase columns:")
print(
    base_df.columns.tolist()
)

print("\nPrediction columns:")
print(
    pred_df.columns.tolist()
)


required_base = [
    "id",
    "original_script"
]

for col in required_base:

    if col not in base_df.columns:

        raise ValueError(
            f"Missing base column: {col}"
        )


required_predictions = [
    "id",
    "true_label",
    "condition",
    "probability_hate",
    "probability_non_hate"
]

for col in required_predictions:

    if col not in pred_df.columns:

        raise ValueError(
            f"Missing prediction column: {col}"
        )


# ============================================================
# ORIGINAL CONDITION
# ============================================================

original = pred_df[
    pred_df["condition"] == "original"
].copy()

original = original.sort_values(
    "id"
).reset_index(drop=True)


# ============================================================
# ADD ORIGINAL SCRIPT
# ============================================================

script_info = base_df[
    ["id", "original_script"]
].drop_duplicates(
    subset=["id"]
)

original = original.merge(
    script_info,
    on="id",
    how="left"
)


# ============================================================
# CHECK SCRIPT INFORMATION
# ============================================================

missing_scripts = (
    original["original_script"]
    .isna()
    .sum()
)

print(
    "\nMissing original_script values:",
    missing_scripts
)

if missing_scripts > 0:

    raise ValueError(
        "Some XLM-R predictions could not be matched "
        "with original script information."
    )


# ============================================================
# SCRIPT DISTRIBUTION
# ============================================================

print("\nOriginal script distribution:")

print(
    original[
        "original_script"
    ].value_counts()
)


# ============================================================
# LABELS
# ============================================================

y = np.array([
    CLASS_TO_INDEX[str(label)]
    for label in original["true_label"]
])


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = original[
    [
        "probability_hate",
        "probability_non_hate"
    ]
].values.astype(float)


# ============================================================
# CALIBRATION / FINAL EVALUATION SPLIT
# ============================================================

indices = np.arange(
    len(original)
)

calibration_indices, evaluation_indices = (
    train_test_split(
        indices,
        test_size=400,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print("\nData split:")

print(
    "Calibration:",
    len(calibration_indices)
)

print(
    "Final evaluation:",
    len(evaluation_indices)
)


# ============================================================
# GLOBAL CALIBRATION
# ============================================================

cal_probs = probabilities[
    calibration_indices
]

cal_labels = y[
    calibration_indices
]

cal_scripts = original[
    "original_script"
].values[
    calibration_indices
]


# True-class probabilities

true_probs = np.array([

    cal_probs[i, cal_labels[i]]

    for i in range(
        len(cal_labels)
    )

])


# Nonconformity

global_scores = (
    1.0 - true_probs
)


global_threshold = conformal_quantile(
    global_scores,
    TARGET_COVERAGE
)


print("\nGlobal threshold:")

print(
    f"{global_threshold:.6f}"
)


# ============================================================
# SCRIPT-SPECIFIC THRESHOLDS
# ============================================================

script_thresholds = {}

script_sample_counts = {}

unique_scripts = sorted(
    original[
        "original_script"
    ].unique()
)


print("\n" + "=" * 80)
print("SCRIPT-SPECIFIC CALIBRATION")
print("=" * 80)


for script in unique_scripts:

    mask = (
        cal_scripts == script
    )

    script_scores = (
        global_scores[mask]
    )

    count = len(
        script_scores
    )

    script_sample_counts[
        script
    ] = count

    print(
        f"\nScript: {script}"
    )

    print(
        f"Calibration samples: {count}"
    )

    # Small groups use global calibration
    if count < MIN_GROUP_CALIBRATION:

        threshold = global_threshold

        print(
            "Using GLOBAL threshold "
            "(insufficient group samples)."
        )

    else:

        threshold = conformal_quantile(
            script_scores,
            TARGET_COVERAGE
        )

        print(
            f"Script-aware threshold: "
            f"{threshold:.6f}"
        )

    script_thresholds[
        script
    ] = threshold


# ============================================================
# EVALUATE ALL CONDITIONS
# ============================================================

results = []


for condition in CONDITIONS:

    print("\n" + "-" * 70)

    print(
        "Condition:",
        condition
    )

    print("-" * 70)


    condition_df = pred_df[
        pred_df["condition"] == condition
    ].copy()


    condition_df = condition_df.sort_values(
        "id"
    ).reset_index(drop=True)


    # Add original script
    condition_df = condition_df.merge(
        script_info,
        on="id",
        how="left"
    )


    if len(condition_df) != 800:

        raise ValueError(
            f"{condition}: expected 800 rows, "
            f"found {len(condition_df)}"
        )


    condition_probs = condition_df[
        [
            "probability_hate",
            "probability_non_hate"
        ]
    ].values.astype(float)


    condition_labels = np.array([

        CLASS_TO_INDEX[str(label)]

        for label in condition_df[
            "true_label"
        ]

    ])


    condition_scripts = (
        condition_df[
            "original_script"
        ].values
    )


    # Same final evaluation indices
    # for every condition

    eval_probs = condition_probs[
        evaluation_indices
    ]

    eval_labels = condition_labels[
        evaluation_indices
    ]

    eval_scripts = condition_scripts[
        evaluation_indices
    ]


    prediction_sets = []


    for probs, script in zip(
        eval_probs,
        eval_scripts
    ):

        threshold = script_thresholds[
            script
        ]


        pred_set = create_prediction_set(
            probs,
            threshold
        )


        prediction_sets.append(
            pred_set
        )


    metrics = evaluate_prediction_sets(
        prediction_sets,
        eval_labels
    )


    metrics[
        "Condition"
    ] = condition

    metrics[
        "Target Coverage"
    ] = TARGET_COVERAGE

    metrics[
        "Global Threshold"
    ] = global_threshold

    metrics[
        "Calibration Samples"
    ] = len(calibration_indices)

    metrics[
        "Final Evaluation Samples"
    ] = len(evaluation_indices)


    results.append(
        metrics
    )


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
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df[
    [
        "Condition",
        "Target Coverage",
        "Coverage",
        "Average Prediction Set Size",
        "Singleton Sets",
        "Full Sets",
        "Empty Sets",
        "Human Review Rate",
        "Global Threshold",
        "Calibration Samples",
        "Final Evaluation Samples"
    ]
]


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {

    "method":
        "Script-Aware Split Conformal Prediction",

    "model":
        "XLM-R",

    "target_coverage":
        TARGET_COVERAGE,

    "random_state":
        RANDOM_STATE,

    "calibration_samples":
        len(calibration_indices),

    "final_evaluation_samples":
        len(evaluation_indices),

    "minimum_group_calibration":
        MIN_GROUP_CALIBRATION,

    "global_threshold":
        global_threshold,

    "script_thresholds":
        script_thresholds,

    "script_calibration_sample_counts":
        script_sample_counts,

    "small_group_policy":
        "Use global threshold",

    "conditions":
        CONDITIONS
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
print("SCRIPT-AWARE CONFORMAL PREDICTION COMPLETE")
print("=" * 80)

print("\nFinal Results:")

print(
    results_df.to_string(
        index=False
    )
)

print("\nScript thresholds:")

for script, threshold in (
    script_thresholds.items()
):

    print(
        f"{script}: "
        f"{threshold:.6f}"
    )


print("\nSaved:")

print(
    OUTPUT_FILE
)

print(
    CONFIG_FILE
)

print("\nDONE.")