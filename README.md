# SCRIPT-CERT

## Risk-Controlled Hate-Speech Detection for Telugu–English Code-Mixed Text

SCRIPT-CERT studies the robustness and reliability of hate-speech detection for Telugu–English code-mixed text under script, transliteration, and spelling shifts.

### Research Objective

The project evaluates whether hate-speech classifiers maintain reliable predictions when the same text is represented using different scripts or spelling variations.

The evaluation considers:

- Original Telugu–English code-mixed text
- Romanized text
- Telugu-script transformations
- Mixed-script text
- Mild spelling noise
- Severe spelling noise

### Models

The project currently includes:

- TF-IDF + Logistic Regression baseline
- MuRIL-based classifier
- XLM-R based classifier

The transformer experiments use frozen representations with downstream classification to keep the experiments computationally practical.

### Evaluation

The project evaluates:

- Accuracy
- Balanced Accuracy
- Macro-F1
- Prediction flip rate
- Confidence changes
- Expected Calibration Error (ECE)
- Brier Score
- Negative Log-Likelihood (NLL)
- Temperature scaling
- Conformal prediction
- Script-aware conformal prediction
- Risk-coverage analysis
- Human-review rate

### Project Structure

```text
SCRIPT_CERT/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── script_shift/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── transformations/
│   ├── calibration/
│   ├── conformal/
│   └── evaluation/
│
├── experiments/
│   ├── baselines/
│   ├── script_shift/
│   ├── calibration/
│   └── conformal/
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── predictions/
│
├── requirements.txt
├── project_structure.txt
└── README.md
