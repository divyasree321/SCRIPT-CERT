import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("results/tables/model_comparison_script_shift.csv")
OUTPUT_DIR = Path("results/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nMODEL COMPARISON PLOT")
print("=" * 70)

print(df.to_string(index=False))


# ============================================================
# CONDITION ORDER
# ============================================================

condition_order = [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise"
]

condition_labels = [
    "Original",
    "Romanized",
    "Telugu Script",
    "Mixed Script",
    "Mild Noise",
    "Severe Noise"
]


# ============================================================
# PLOT 1: ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

for model in ["MuRIL", "XLM-R"]:

    model_df = (
        df[df["Model"] == model]
        .set_index("Condition")
        .reindex(condition_order)
    )

    plt.plot(
        condition_labels,
        model_df["Accuracy"],
        marker="o",
        linewidth=2,
        label=model
    )

plt.xlabel("SCRIPT-SHIFT Condition")
plt.ylabel("Accuracy")
plt.title("Model Accuracy under SCRIPT-SHIFT Conditions")

plt.xticks(rotation=25)
plt.ylim(0.40, 0.85)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

accuracy_file = OUTPUT_DIR / "model_accuracy_script_shift.png"

plt.savefig(
    accuracy_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(accuracy_file)


# ============================================================
# PLOT 2: MACRO-F1
# ============================================================

plt.figure(figsize=(10, 6))

for model in ["MuRIL", "XLM-R"]:

    model_df = (
        df[df["Model"] == model]
        .set_index("Condition")
        .reindex(condition_order)
    )

    plt.plot(
        condition_labels,
        model_df["Macro-F1"],
        marker="o",
        linewidth=2,
        label=model
    )

plt.xlabel("SCRIPT-SHIFT Condition")
plt.ylabel("Macro-F1")
plt.title("Macro-F1 under SCRIPT-SHIFT Conditions")

plt.xticks(rotation=25)
plt.ylim(0.25, 0.85)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

f1_file = OUTPUT_DIR / "model_macro_f1_script_shift.png"

plt.savefig(
    f1_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(f1_file)


# ============================================================
# PLOT 3: PREDICTION FLIP RATE
# ============================================================

plt.figure(figsize=(10, 6))

for model in ["MuRIL", "XLM-R"]:

    model_df = (
        df[df["Model"] == model]
        .set_index("Condition")
        .reindex(condition_order)
    )

    plt.plot(
        condition_labels,
        model_df["Prediction Flip Rate"],
        marker="o",
        linewidth=2,
        label=model
    )

plt.xlabel("SCRIPT-SHIFT Condition")
plt.ylabel("Prediction Flip Rate")
plt.title("Prediction Flip Rate under SCRIPT-SHIFT")

plt.xticks(rotation=25)
plt.ylim(0, 0.65)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

flip_file = OUTPUT_DIR / "model_prediction_flip_rate_script_shift.png"

plt.savefig(
    flip_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(flip_file)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PLOTS CREATED SUCCESSFULLY")
print("=" * 70)

print(f"\n1. {accuracy_file}")
print(f"2. {f1_file}")
print(f"3. {flip_file}")