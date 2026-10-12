# SCRIPT-CERT

## Risk-Controlled Hate-Speech Detection for Telugu–English Code-Mixed Text Under Transliteration and Spelling Shifts

SCRIPT-CERT is a research project that investigates the robustness and reliability of hate-speech detection models for Telugu–English code-mixed text when the same or similar content is expressed under different writing scripts, transliteration patterns, and spelling variations.

The project focuses not only on classification accuracy, but also on **prediction stability, confidence calibration, selective prediction, and conformal prediction**. The goal is to study when a hate-speech classifier changes its prediction under script or spelling shifts and how uncertainty-aware methods can support safer human review.

---

## Research Objective

Hate-speech detection systems for multilingual and code-mixed social-media text can be sensitive to changes in writing style.

For example, Telugu content may appear as:

- Telugu native script
- Romanized Telugu
- Telugu–English code-mixed text
- Mixed Telugu and Roman scripts
- Text containing spelling variations
- Text containing mild or severe character-level noise

SCRIPT-CERT evaluates whether classification predictions remain stable under these transformations.

The study therefore considers three related questions:

1. **Robustness:** How much do hate-speech classifiers degrade under script and spelling shifts?
2. **Reliability:** Do model confidence scores remain calibrated when the input representation changes?
3. **Risk control:** Can conformal prediction and selective prediction identify uncertain cases that should be sent for human review?

---

# Dataset

The project uses the Telugu hate/offensive-language dataset associated with the **DravidianLangTech@EACL 2024 shared task**.

The locally available dataset contains:

- Training samples: **4,000**
- Locked test samples: **499**
- Training classes:
  - `non-hate`: 2,061
  - `hate`: 1,939

The training data is used for development and validation.

The official test set is kept separate from the development workflow and is treated as a locked evaluation resource.

## Dataset Files

```text
data/
├── raw/
│   ├── training_data_telugu-hate.xlsx
│   └── telugu-hate-speech-test.xlsx
│
├── processed/
│   └── train_clean.csv
│
└── script_shift/
    ├── script_shift_base.csv
    ├── script_shift_base_fixed.csv
    ├── validation_script_shift.csv
    └── validation_script_shift_full.csv