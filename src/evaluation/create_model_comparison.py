import os
import pandas as pd

# ============================================================
# FILES
# ============================================================

MURIL_FILE = "results/tables/muril_script_shift_results.csv"
MURIL_FLIP_FILE = "results/tables/muril_prediction_flip_rates.csv"
XLMR_FILE = "results/tables/xlmr_script_shift_results.csv"

OUTPUT_FILE = "results/tables/model_comparison_script_shift.csv"

# ============================================================
# LOAD RESULTS
# ============================================================

muril = pd.read_csv(MURIL_FILE)
muril_flip = pd.read_csv(MURIL_FLIP_FILE)
xlmr = pd.read_csv(XLMR_FILE)

# Clean column names
muril.columns = muril.columns.str.strip()
muril_flip.columns = muril_flip.columns.str.strip()
xlmr.columns = xlmr.columns.str.strip()

print("MuRIL columns:")
print(muril.columns.tolist())

print("\nMuRIL flip columns:")
print(muril_flip.columns.tolist())

print("\nXLM-R columns:")
print(xlmr.columns.tolist())

# ============================================================
# STANDARDIZE MuRIL COLUMN NAMES
# ============================================================

if "Macro F1" in muril.columns:
    muril = muril.rename(
        columns={"Macro F1": "Macro-F1"}
    )

# ============================================================
# ADD MODEL NAMES
# ============================================================

muril["Model"] = "MuRIL"
xlmr["Model"] = "XLM-R"

# ============================================================
# ADD MuRIL FLIP RATE
# ============================================================

muril_flip = muril_flip.rename(
    columns={
        "Flip Rate": "Prediction Flip Rate"
    }
)

muril = muril.merge(
    muril_flip[
        [
            "Condition",
            "Prediction Flip Rate"
        ]
    ],
    on="Condition",
    how="left"
)

# ============================================================
# CHECK XLM-R FLIP RATE
# ============================================================

if "Prediction Flip Rate" not in xlmr.columns:
    raise ValueError(
        "XLM-R results do not contain Prediction Flip Rate."
    )

# ============================================================
# COMMON COLUMNS
# ============================================================

columns = [
    "Model",
    "Condition",
    "Accuracy",
    "Balanced Accuracy",
    "Macro-F1",
    "Macro Precision",
    "Macro Recall",
    "Mean Confidence",
    "Prediction Flip Rate"
]

muril = muril[columns]
xlmr = xlmr[columns]

# ============================================================
# COMBINE
# ============================================================

combined = pd.concat(
    [muril, xlmr],
    ignore_index=True
)

# ============================================================
# ORDER CONDITIONS
# ============================================================

condition_order = [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise"
]

model_order = [
    "MuRIL",
    "XLM-R"
]

combined["Condition"] = pd.Categorical(
    combined["Condition"],
    categories=condition_order,
    ordered=True
)

combined["Model"] = pd.Categorical(
    combined["Model"],
    categories=model_order,
    ordered=True
)

combined = combined.sort_values(
    ["Condition", "Model"]
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True
)

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

print(
    combined.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 100)
print("MISSING VALUE CHECK")
print("=" * 100)

print(combined.isna().sum())

# ============================================================
# FINAL CHECK
# ============================================================

if combined.isna().sum().sum() == 0:
    print("\nSUCCESS: No missing values in comparison table.")
else:
    print("\nWARNING: Missing values detected.")

print("\nSaved:")
print(OUTPUT_FILE)