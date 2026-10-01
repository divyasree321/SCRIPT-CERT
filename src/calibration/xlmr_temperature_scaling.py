import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_FILE = "results/predictions/xlmr_script_shift_predictions.csv"
OUTPUT_FILE = "results/tables/xlmr_temperature_scaling_final.csv"

RANDOM_STATE = 42


# ============================================================
# ECE FUNCTION
# ============================================================

def calculate_ece(probabilities, y_true, n_bins=10):
    """
    Expected Calibration Error for binary classification.
    """

    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    predictions = (probabilities >= 0.5).astype(int)

    confidences = np.maximum(probabilities, 1 - probabilities)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (confidences >= bin_edges[i]) &
                (confidences <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidences >= bin_edges[i]) &
                (confidences < bin_edges[i + 1])
            )

        if np.sum(mask) == 0:
            continue

        bin_accuracy = np.mean(correct[mask])
        bin_confidence = np.mean(confidences[mask])
        bin_fraction = np.mean(mask)

        ece += abs(bin_accuracy - bin_confidence) * bin_fraction

    return ece


# ============================================================
# TEMPERATURE SCALING
# ============================================================

def apply_temperature(probabilities, temperature):

    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

    logits = np.log(
        probabilities / (1.0 - probabilities)
    )

    scaled_logits = logits / temperature

    calibrated_probabilities = 1.0 / (
        1.0 + np.exp(-scaled_logits)
    )

    return calibrated_probabilities


def find_temperature(probabilities, y_true):

    temperatures = np.linspace(0.05, 5.0, 496)

    best_temperature = 1.0
    best_nll = float("inf")

    for temperature in temperatures:

        calibrated = apply_temperature(
            probabilities,
            temperature
        )

        nll = log_loss(
            y_true,
            calibrated,
            labels=[0, 1]
        )

        if nll < best_nll:

            best_nll = nll
            best_temperature = temperature

    return best_temperature


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("XLM-R TEMPERATURE SCALING")
print("=" * 70)

df = pd.read_csv(PREDICTION_FILE)

print("\nLoaded:")
print(PREDICTION_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# USE ORIGINAL CONDITION ONLY FOR CALIBRATION SPLIT
# ============================================================

original = df[
    df["condition"] == "original"
].copy()

print("\nOriginal condition:")
print(original.shape)


# ============================================================
# LABEL ENCODING
# ============================================================

label_map = {
    "hate": 1,
    "non-hate": 0
}

original["y"] = original["true_label"].map(label_map)

if original["y"].isna().any():
    raise ValueError("Unknown labels found.")

original["prob_hate"] = original["probability_hate"].astype(float)


# ============================================================
# CALIBRATION / FINAL EVALUATION SPLIT
# ============================================================

indices = np.arange(len(original))

cal_idx, eval_idx = train_test_split(
    indices,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=original["y"]
)

calibration = original.iloc[cal_idx].copy()
evaluation = original.iloc[eval_idx].copy()

print("\nCalibration samples:", len(calibration))
print("Final evaluation samples:", len(evaluation))


# ============================================================
# FIND TEMPERATURE
# ============================================================

temperature = find_temperature(
    calibration["prob_hate"].values,
    calibration["y"].values
)

print("\nOptimal temperature:")
print(f"{temperature:.4f}")


# ============================================================
# ORIGINAL EVALUATION PROBABILITIES
# ============================================================

y_true = evaluation["y"].values
prob_before = evaluation["prob_hate"].values


# ============================================================
# CALIBRATED PROBABILITIES
# ============================================================

prob_after = apply_temperature(
    prob_before,
    temperature
)


# ============================================================
# METRICS
# ============================================================

nll_before = log_loss(
    y_true,
    prob_before,
    labels=[0, 1]
)

nll_after = log_loss(
    y_true,
    prob_after,
    labels=[0, 1]
)


brier_before = np.mean(
    (prob_before - y_true) ** 2
)

brier_after = np.mean(
    (prob_after - y_true) ** 2
)


ece_before = calculate_ece(
    prob_before,
    y_true
)

ece_after = calculate_ece(
    prob_after,
    y_true
)


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Model": "XLM-R",
        "Evaluation": "Original",
        "Temperature": temperature,
        "NLL": nll_before,
        "Brier Score": brier_before,
        "ECE": ece_before
    },
    {
        "Model": "XLM-R",
        "Evaluation": "Temperature Scaled",
        "Temperature": temperature,
        "NLL": nll_after,
        "Brier Score": brier_after,
        "ECE": ece_after
    }
])


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION RESULTS")
print("=" * 70)

print(results.to_string(index=False))

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDONE.")