
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# SCRIPT-CERT Research Dashboard
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

st.set_page_config(
    page_title="SCRIPT-CERT | Research Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SCRIPT-CERT")
st.subheader(
    "Risk-Controlled Hate-Speech Detection for "
    "Telugu–English Code-Mixed Text"
)
st.caption(
    "Research dashboard | Script robustness | "
    "Calibration | Conformal prediction"
)

st.info(
    "This dashboard displays saved experiment results. "
    "Its charts describe the recorded experiments, not "
    "predictions from a newly trained model."
)


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
        st.warning(f"Result file not found: {filename}")
        return None

    st.dataframe(df, use_container_width=True)
    st.download_button(
        label=f"Download {title} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )
    return df


# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Model Comparison",
        "Script Robustness",
        "Calibration",
        "Conformal Prediction",
        "Saved Figures",
        "Result Explorer",
    ],
)

# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":
    st.header("Project Overview")

    st.markdown("""
    SCRIPT-CERT studies whether hate-speech classification remains
    reliable when Telugu–English code-mixed comments are represented
    using different scripts and spelling variations.

    **Research workflow**
    1. Classify comments as hate or non-hate.
    2. Evaluate predictions under script and spelling shifts.
    3. Measure prediction instability.
    4. Examine probability calibration.
    5. Estimate prediction-set coverage and review workload.
    """)

    st.subheader("Headline result from saved experiments")

    df = load_csv("xlmr_script_shift_results.csv")

    if df is not None:
        st.write("XLM-R experiment results")
        st.dataframe(df, use_container_width=True)

        st.caption(
            "The original-condition accuracy previously recorded "
            "for this experiment was 78.75%. Confirm the exact "
            "column values in the table before reporting them."
        )
    else:
        st.warning(
            "Could not find xlmr_script_shift_results.csv "
            "inside results/tables."
        )

    st.subheader("Project information")
    a, b, c = st.columns(3)
    a.metric("Task", "Hate / Non-hate")
    b.metric("Language", "Telugu–English")
    c.metric("Focus", "Robustness & reliability")

# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

elif page == "Model Comparison":
    st.header("Model Comparison")

    df = show_table(
        "model_comparison_script_shift.csv",
        "Model comparison results",
    )

    if df is not None:
        st.write("Available columns:", ", ".join(df.columns))

        numeric = df.select_dtypes(include="number").columns.tolist()
        if numeric:
            metric = st.selectbox("Choose a numeric metric", numeric)
            category_cols = [
                c for c in df.columns
                if c not in numeric and df[c].nunique() > 1
            ]

            if category_cols:
                category = st.selectbox(
                    "Group results by", category_cols
                )
                chart = px.bar(
                    df,
                    x=category,
                    y=metric,
                    color=category,
                    title=f"{metric} comparison",
                    text_auto=".3f",
                )
                st.plotly_chart(chart, use_container_width=True)

# --------------------------------------------------
# Script Robustness
# --------------------------------------------------

elif page == "Script Robustness":
    st.header("SCRIPT-SHIFT Robustness")

    df = show_table(
        "xlmr_script_shift_results.csv",
        "XLM-R results by text condition",
    )

    show_table(
        "xlmr_prediction_flip_rates.csv",
        "XLM-R prediction flip rates",
    )

    show_table(
        "xlmr_confidence_shift.csv",
        "XLM-R confidence changes",
    )

    if df is not None:
        numeric = df.select_dtypes(include="number").columns.tolist()
        category_cols = [
            c for c in df.columns
            if c not in numeric and df[c].nunique() > 1
        ]

        if numeric and category_cols:
            metric = st.selectbox(
                "Metric to visualize", numeric, key="robust_metric"
            )
            category = st.selectbox(
                "Text condition column",
                category_cols,
                key="robust_category",
            )

            st.plotly_chart(
                px.bar(
                    df,
                    x=category,
                    y=metric,
                    title=f"{metric} across text conditions",
                    color=category,
                    text_auto=".3f",
                ),
                use_container_width=True,
            )

# --------------------------------------------------
# Calibration
# --------------------------------------------------

elif page == "Calibration":
    st.header("Probability Calibration")

    show_table(
        "temperature_scaling_final.csv",
        "TF-IDF temperature scaling results",
    )

    show_table(
        "xlmr_temperature_scaling_final.csv",
        "XLM-R temperature scaling results",
    )

    st.markdown("""
    **ECE:** Expected Calibration Error.

    **Brier score:** Measures the accuracy of predicted probabilities.

    **NLL:** Negative Log-Likelihood; lower values are generally better.
    """)

# --------------------------------------------------
# Conformal Prediction
# --------------------------------------------------

elif page == "Conformal Prediction":
    st.header("Conformal Prediction & Review Workload")

    show_table(
        "conformal_prediction_final.csv",
        "TF-IDF conformal prediction",
    )

    show_table(
        "xlmr_conformal_prediction_final.csv",
        "XLM-R conformal prediction",
    )

    show_table(
        "xlmr_script_aware_conformal_final.csv",
        "XLM-R script-aware conformal prediction",
    )

    st.caption(
        "Review rate is an estimated workload under the configured "
        "prediction-set rule. It does not represent actual human review."
    )

# --------------------------------------------------
# Saved Figures
# --------------------------------------------------

elif page == "Saved Figures":
    st.header("Saved Experiment Figures")

    if FIGURES.exists():
        images = sorted(
            p for p in FIGURES.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

        if images:
            names = [p.name for p in images]
            selected = st.selectbox("Choose a figure", names)
            image_path = FIGURES / selected
            st.image(str(image_path), caption=selected, use_container_width=True)
        else:
            st.info("No image files were found in results/figures.")
    else:
        st.warning("The results/figures folder was not found.")

# --------------------------------------------------
# Result Explorer
# --------------------------------------------------

elif page == "Result Explorer":
    st.header("Explore All Saved Result Tables")

    if TABLES.exists():
        files = sorted(TABLES.glob("*.csv"))

        if files:
            selected = st.selectbox(
                "Select a CSV file",
                files,
                format_func=lambda p: p.name,
            )

            try:
                df = pd.read_csv(selected)
                st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "Download selected CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=selected.name,
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")
        else:
            st.warning("No CSV result files were found.")
    else:
        st.error("Could not find the results/tables folder.")

st.divider()
st.caption(
    "SCRIPT-CERT research prototype | Results are from saved "
    "experiments and should be interpreted with their evaluation splits."
)
