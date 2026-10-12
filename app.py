from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "live_model"
ENCODER_DIR = MODEL_DIR / "encoder"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"

st.set_page_config(page_title="SCRIPT-CERT | Research Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ SCRIPT-CERT")
st.subheader("Risk-Controlled Hate-Speech Detection for Telugu–English Code-Mixed Text")
st.caption("Research dashboard | Script robustness | Calibration | Conformal prediction | Live XLM-R prediction")
st.info("Research prototype only. Predictions are experimental outputs, not definitive moderation decisions. Use human review for uncertain or consequential cases.")


@st.cache_data
def load_csv(filename):
    path = TABLES / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def show_table(filename, title):
    st.subheader(title)
    df = load_csv(filename)
    if df is None:
        st.warning(f"Result file not found: results/tables/{filename}")
        return None
    st.dataframe(df, use_container_width=True)
    st.download_button(
        f"Download {title} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )
    return df


@st.cache_resource(show_spinner="Loading the saved XLM-R model...")
def load_live_model():
    required = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "classifier.joblib",
        MODEL_DIR / "label_encoder.joblib",
        ENCODER_DIR / "config.json",
        ENCODER_DIR / "model.safetensors",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing saved model file(s): " + ", ".join(missing))

    with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    classifier = joblib.load(MODEL_DIR / "classifier.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

    # Import inside the function so the rest of the dashboard can load even
    # if the model dependencies need fixing.
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer_source = TOKENIZER_DIR if TOKENIZER_DIR.is_dir() else ENCODER_DIR
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_source), local_files_only=True, use_fast=False
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            str(ENCODER_DIR), local_files_only=True, use_fast=False
        )

    model = AutoModel.from_pretrained(str(ENCODER_DIR), local_files_only=True)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    return classifier, label_encoder, tokenizer, model, config, device


def get_comment_embedding(text, components):
    """Attention-mask-weighted mean pooling for one comment."""
    import torch
    _, _, tokenizer, model, config, device = components
    max_length = int(config.get("max_length", 128))
    encoded = tokenizer(
        [text], padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embedding = (summed / counts).cpu().numpy()
    return embedding


def predict_comment(text, components):
    classifier, label_encoder, _, _, _, _ = components
    embedding = get_comment_embedding(text, components)
    probabilities = classifier.predict_proba(embedding)[0]
    predicted_index = int(classifier.predict(embedding)[0])
    label = str(label_encoder.inverse_transform([predicted_index])[0])
    confidence = float(np.max(probabilities))

    try:
        class_names = [
            str(x) for x in label_encoder.inverse_transform(
                np.asarray(classifier.classes_, dtype=int)
            )
        ]
    except Exception:
        class_names = [f"Class {i}" for i in range(len(probabilities))]
    probability_map = {
        class_names[i] if i < len(class_names) else f"Class {i}": float(probabilities[i])
        for i in range(len(probabilities))
    }
    return label, confidence, probability_map


def review_advice(confidence):
    if confidence < 0.70:
        return "Human review recommended", "Confidence is below 70%; do not make an automatic moderation decision."
    if confidence < 0.85:
        return "Review advised", "The model is moderately confident. Check context before taking action."
    return "Human oversight still advised", "High confidence is not proof of correctness; check context before taking action."


page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Live Prediction", "Model Comparison", "Script Robustness",
     "Calibration", "Conformal Prediction", "Saved Figures", "Result Explorer"],
)

if page == "Overview":
    st.header("Project Overview")
    st.markdown("""
SCRIPT-CERT studies whether Telugu–English code-mixed hate-speech classification
remains reliable under script and spelling changes.

**Research workflow**
1. Classify comments as hate or non-hate.
2. Evaluate predictions under script and spelling shifts.
3. Measure prediction instability and confidence changes.
4. Examine probability calibration.
5. Estimate conformal prediction coverage and review workload.
""")
    df = load_csv("xlmr_script_shift_results.csv")
    st.subheader("Saved XLM-R experiment results")
    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.caption("These are saved evaluation results, not live predictions. Report the evaluation split with each metric.")
    else:
        st.warning("Could not find results/tables/xlmr_script_shift_results.csv.")
    a, b, c = st.columns(3)
    a.metric("Task", "Hate / Non-hate")
    b.metric("Language", "Telugu–English")
    c.metric("Focus", "Robustness & reliability")
    st.info("Choose **Live Prediction** in the sidebar to test a new comment.")

elif page == "Live Prediction":
    st.header("🔎 Live Telugu–English Comment Tester")
    st.write("Enter a code-mixed comment. The saved XLM-R encoder creates an embedding, then the saved Logistic Regression classifier predicts a label.")
    st.caption("XLM-RoBERTa-base with frozen embeddings + Logistic Regression. Inference runs locally on the CPU.")
    with st.form("live_prediction_form"):
        comment = st.text_area(
            "Enter a Telugu–English comment",
            placeholder="Example: Nuv assal manchodivi kaadhu",
            height=120,
            help="Enter one comment at a time. Avoid private information.",
        )
        submitted = st.form_submit_button("Analyze Comment", type="primary", use_container_width=True)

    if submitted:
        if not comment.strip():
            st.warning("Please enter a comment before clicking Analyze Comment.")
        else:
            try:
                components = load_live_model()
                with st.spinner("Analyzing with the saved XLM-R model..."):
                    label, confidence, probabilities = predict_comment(comment.strip(), components)

                normalized = label.lower().replace("_", "-")
                if "non-hate" in normalized or "nonhate" in normalized:
                    st.success(f"Prediction: NON-HATE ({label})")
                elif "hate" in normalized:
                    st.error(f"Prediction: HATE ({label})")
                else:
                    st.info(f"Prediction: {label}")

                col1, col2 = st.columns(2)
                col1.metric("Predicted label", label.replace("_", " ").upper())
                col2.metric("Model confidence", f"{confidence:.2%}")

                advice, explanation = review_advice(confidence)
                st.warning(f"Recommendation: {advice}")
                st.write(explanation)

                st.subheader("Class probability scores")
                prob_df = pd.DataFrame({"Class": list(probabilities.keys()), "Probability": list(probabilities.values())})
                st.dataframe(
                    prob_df.assign(Probability=prob_df["Probability"].map(lambda x: f"{x:.2%}")),
                    use_container_width=True, hide_index=True,
                )
                st.plotly_chart(
                    px.bar(
                        prob_df, x="Class", y="Probability", range_y=[0, 1],
                        text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
                        title="Model probability by class",
                    ),
                    use_container_width=True,
                )
                st.caption("Confidence is the model's maximum predicted probability, not a guarantee of correctness.")
            except Exception as exc:
                st.error("Could not generate a prediction. Check the saved model layout and Python dependencies.")
                st.exception(exc)
                st.markdown("""
**Expected saved model layout**
```text
results/live_model/
├── config.json
├── classifier.joblib
├── label_encoder.joblib
├── encoder/
│   ├── config.json
│   └── model.safetensors
└── tokenizer/  (optional; tokenizer files may instead be in encoder/)
```
Do not rerun training just to fix a path or dependency error.
""")

elif page == "Model Comparison":
    st.header("Model Comparison")
    df = show_table("model_comparison_script_shift.csv", "Model comparison results")
    if df is not None:
        numeric = df.select_dtypes(include="number").columns.tolist()
        categories = [c for c in df.columns if c not in numeric and df[c].nunique(dropna=True) > 1]
        if numeric:
            metric = st.selectbox("Choose a numeric metric", numeric, key="comparison_metric")
            if categories:
                category = st.selectbox("Group results by", categories, key="comparison_category")
                st.plotly_chart(px.bar(df, x=category, y=metric, color=category, title=f"{metric} comparison", text_auto=".3f"), use_container_width=True)

elif page == "Script Robustness":
    st.header("SCRIPT-SHIFT Robustness")
    df = show_table("xlmr_script_shift_results.csv", "XLM-R results by text condition")
    show_table("xlmr_prediction_flip_rates.csv", "XLM-R prediction flip rates")
    show_table("xlmr_confidence_shift.csv", "XLM-R confidence changes")
    if df is not None:
        numeric = df.select_dtypes(include="number").columns.tolist()
        categories = [c for c in df.columns if c not in numeric and df[c].nunique(dropna=True) > 1]
        if numeric and categories:
            metric = st.selectbox("Metric to visualize", numeric, key="robust_metric")
            category = st.selectbox("Text condition column", categories, key="robust_category")
            st.plotly_chart(px.bar(df, x=category, y=metric, color=category, title=f"{metric} across text conditions", text_auto=".3f"), use_container_width=True)

elif page == "Calibration":
    st.header("Probability Calibration")
    show_table("temperature_scaling_final.csv", "TF-IDF temperature scaling results")
    show_table("xlmr_temperature_scaling_final.csv", "XLM-R temperature scaling results")
    st.markdown("""
- **ECE (Expected Calibration Error):** compares confidence with observed accuracy across confidence bins.
- **Brier score:** evaluates probability predictions; lower is generally better.
- **NLL (Negative Log-Likelihood):** penalizes assigning low probability to the true class; lower is generally better.
""")

elif page == "Conformal Prediction":
    st.header("Conformal Prediction & Review Workload")
    show_table("conformal_prediction_final.csv", "TF-IDF conformal prediction")
    show_table("xlmr_conformal_prediction_final.csv", "XLM-R conformal prediction")
    show_table("xlmr_script_aware_conformal_final.csv", "XLM-R script-aware conformal prediction")
    st.caption("Review rate is estimated from prediction-set rules; it does not represent actual human review.")

elif page == "Saved Figures":
    st.header("Saved Experiment Figures")
    if FIGURES.exists():
        images = sorted(p for p in FIGURES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if images:
            selected = st.selectbox("Choose a figure", images, format_func=lambda p: p.name)
            st.image(str(selected), caption=selected.name, use_container_width=True)
        else:
            st.info("No image files were found in results/figures.")
    else:
        st.warning("The results/figures folder was not found.")

elif page == "Result Explorer":
    st.header("Explore All Saved Result Tables")
    if TABLES.exists():
        files = sorted(TABLES.glob("*.csv"))
        if files:
            selected = st.selectbox("Select a CSV file", files, format_func=lambda p: p.name)
            try:
                df = pd.read_csv(selected)
                st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download selected CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=selected.name,
                    mime="text/csv",
                    key=f"explorer_{selected.name}",
                )
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")
        else:
            st.warning("No CSV result files were found.")
    else:
        st.error("Could not find the results/tables folder.")

st.divider()
st.caption("SCRIPT-CERT research prototype | Saved experiment results are separate from live predictions. Interpret metrics with their evaluation splits.")
