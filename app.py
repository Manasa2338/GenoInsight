import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GenoInsight",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "results" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
SELECTED_FEATURES_PATH = (
    BASE_DIR / "data" / "processed" / "selected_features.csv"
)
CLASS_MAPPING_PATH = (
    BASE_DIR / "results" / "class_mapping.csv"
)

RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# SESSION STATE
# ============================================================

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "X_original" not in st.session_state:
    st.session_state.X_original = None

if "X_selected" not in st.session_state:
    st.session_state.X_selected = None

if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    if not SCALER_PATH.exists():
        return None

    return joblib.load(SCALER_PATH)


# ============================================================
# LOAD SELECTED GENES
# ============================================================

@st.cache_data
def load_selected_genes():

    if not SELECTED_FEATURES_PATH.exists():
        return []

    df = pd.read_csv(SELECTED_FEATURES_PATH)

    if "gene" not in df.columns:
        return []

    return df["gene"].astype(str).tolist()


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

@st.cache_data
def load_classes():

    if not CLASS_MAPPING_PATH.exists():
        return ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]

    df = pd.read_csv(CLASS_MAPPING_PATH)

    if "Class" in df.columns:
        return df["Class"].astype(str).tolist()

    return ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]


# ============================================================
# LOAD OBJECTS
# ============================================================

model = load_model()
scaler = load_scaler()
selected_genes = load_selected_genes()
classes = load_classes()


# ============================================================
# HEADER
# ============================================================

st.title("🧬 GenoInsight")

st.subheader(
    "Explainable AI for RNA-Seq Cancer Prediction"
)

st.write(
    """
GenoInsight uses machine learning to predict cancer type from
RNA-Seq gene expression data and provides explainable predictions
using SHAP.
"""
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("📊 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Selected Genes",
        "5,000"
    )

with col2:
    st.metric(
        "Cancer Classes",
        "5"
    )

with col3:
    st.metric(
        "Test Accuracy",
        "99.17%"
    )

with col4:
    st.metric(
        "Test Macro F1",
        "99.29%"
    )


st.info(
    "Model: Elastic Net Logistic Regression"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if model is None:

    st.error(
        f"Model not found:\n{MODEL_PATH}"
    )
    st.stop()


if scaler is None:

    st.error(
        f"Scaler not found:\n{SCALER_PATH}"
    )
    st.stop()


if len(selected_genes) == 0:

    st.error(
        f"Selected genes not found:\n"
        f"{SELECTED_FEATURES_PATH}"
    )
    st.stop()


st.success(
    "✓ Model, scaler and selected genes loaded."
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📁 Upload RNA-Seq Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    key="rna_upload"
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    # Process ONLY when a new file is uploaded
    if (
        st.session_state.uploaded_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Loading and processing RNA-Seq dataset..."
        ):

            try:

                df = pd.read_csv(uploaded_file)

                # --------------------------------------------
                # SAVE ORIGINAL DATA
                # --------------------------------------------

                st.session_state.X_original = df.copy()

                # --------------------------------------------
                # REMOVE SAMPLE ID COLUMN
                # --------------------------------------------

                if "Unnamed: 0" in df.columns:
                    expression_df = df.drop(
                        columns=["Unnamed: 0"]
                    )
                else:
                    expression_df = df.copy()

                # --------------------------------------------
                # CHECK SELECTED GENES
                # --------------------------------------------

                missing_genes = [
                    gene
                    for gene in selected_genes
                    if gene not in expression_df.columns
                ]

                if len(missing_genes) > 0:

                    st.error(
                        f"❌ {len(missing_genes)} selected genes "
                        "are missing from the uploaded dataset."
                    )

                    st.write(
                        "First missing genes:"
                    )

                    st.write(
                        missing_genes[:20]
                    )

                    st.session_state.data_loaded = False
                    st.stop()

                # --------------------------------------------
                # SELECT 5,000 GENES
                # --------------------------------------------

                X_selected = expression_df[
                    selected_genes
                ].copy()

                # --------------------------------------------
                # CONVERT TO NUMERIC
                # --------------------------------------------

                X_selected = X_selected.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                # --------------------------------------------
                # CHECK MISSING VALUES
                # --------------------------------------------

                if X_selected.isnull().sum().sum() > 0:

                    st.error(
                        "❌ Non-numeric or missing values "
                        "were found in selected genes."
                    )

                    st.stop()

                # --------------------------------------------
                # SCALE USING TRAINING SCALER
                # --------------------------------------------

                X_scaled_array = scaler.transform(
                    X_selected
                )

                X_scaled = pd.DataFrame(
                    X_scaled_array,
                    columns=selected_genes,
                    index=X_selected.index
                )

                # --------------------------------------------
                # PREDICTION
                # --------------------------------------------

                predictions_encoded = model.predict(
                    X_scaled
                )

                # --------------------------------------------
                # CONVERT NUMERIC PREDICTIONS TO NAMES
                # --------------------------------------------

                prediction_names = []

                for prediction in predictions_encoded:

                    prediction = int(prediction)

                    if prediction < len(classes):

                        prediction_names.append(
                            classes[prediction]
                        )

                    else:

                        prediction_names.append(
                            str(prediction)
                        )

                # --------------------------------------------
                # PROBABILITIES
                # --------------------------------------------

                probabilities = model.predict_proba(
                    X_scaled
                )

                # --------------------------------------------
                # SAVE EVERYTHING TO SESSION STATE
                # --------------------------------------------

                st.session_state.X_selected = X_selected

                st.session_state.X_scaled = X_scaled

                st.session_state.predictions = (
                    prediction_names
                )

                st.session_state.probabilities = (
                    probabilities
                )

                st.session_state.uploaded_name = (
                    uploaded_file.name
                )

                st.session_state.data_loaded = True

                st.session_state.sample_index = 0

            except Exception as e:

                st.error(
                    f"Error while processing dataset:\n{e}"
                )

                st.session_state.data_loaded = False


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.data_loaded:

    X_original = st.session_state.X_original
    X_selected = st.session_state.X_selected
    predictions = st.session_state.predictions
    probabilities = st.session_state.probabilities


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.header("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Samples",
            X_selected.shape[0]
        )

    with col2:
        st.metric(
            "Original Columns",
            X_original.shape[1]
        )

    with col3:
        st.metric(
            "Required Genes",
            len(selected_genes)
        )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.header("👀 Dataset Preview")

    st.dataframe(
        X_original.head(10),
        use_container_width=True
    )


    # ========================================================
    # GENE VALIDATION
    # ========================================================

    st.header("🧬 Gene Validation")

    missing = [
        gene
        for gene in selected_genes
        if gene not in X_original.columns
    ]

    if len(missing) == 0:

        st.success(
            "✓ All 5,000 selected genes are present."
        )

    else:

        st.error(
            f"❌ Missing genes: {len(missing)}"
        )


    # ========================================================
    # NUMERIC VALIDATION
    # ========================================================

    if (
        X_selected.isnull().sum().sum()
        == 0
    ):

        st.success(
            "✓ All gene expression values are numeric."
        )


    # ========================================================
    # CANCER PREDICTION
    # ========================================================

    st.header("🔬 Cancer Prediction")

    st.success(
        f"✓ Predictions completed for "
        f"{len(predictions)} samples."
    )


    # ========================================================
    # PREDICTION RESULTS TABLE
    # ========================================================

    st.header("🎯 Prediction Results")

    result_df = pd.DataFrame({
        "Sample": [
            f"Sample {i}"
            for i in range(len(predictions))
        ],
        "Predicted Cancer": predictions
    })

    st.dataframe(
        result_df,
        use_container_width=True,
        height=300
    )


    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.header("📊 Prediction Summary")

    prediction_counts = (
        pd.Series(predictions)
        .value_counts()
        .reindex(
            classes,
            fill_value=0
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        prediction_counts.index,
        prediction_counts.values
    )

    ax.set_xlabel(
        "Cancer Type"
    )

    ax.set_ylabel(
        "Number of Samples"
    )

    ax.set_title(
        "Predicted Cancer Distribution"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )


    # ========================================================
    # INDIVIDUAL SAMPLE ANALYSIS
    # ========================================================

    st.header(
        "🔬 Individual Sample Explanation"
    )

    sample_options = list(
        range(len(predictions))
    )

    selected_sample = st.selectbox(
        "Select a sample",
        sample_options,
        index=min(
            st.session_state.sample_index,
            len(sample_options) - 1
        ),
        format_func=lambda x: f"Sample {x}",
        key="individual_sample_selector"
    )

    # IMPORTANT:
    # This changes only session state.
    # The uploaded data and predictions remain stored.

    st.session_state.sample_index = selected_sample


    # ========================================================
    # SELECTED SAMPLE INFORMATION
    # ========================================================

    selected_prediction = predictions[
        selected_sample
    ]

    st.subheader(
        f"Sample {selected_sample}"
    )

    st.write(
        f"**Predicted Cancer:** "
        f"{selected_prediction}"
    )


    # ========================================================
    # PROBABILITY TABLE
    # ========================================================

    st.subheader(
        "🎯 Prediction Probabilities"
    )

    sample_probabilities = probabilities[
        selected_sample
    ]

    probability_df = pd.DataFrame({
        "Cancer": classes,
        "Probability": sample_probabilities
    })

    probability_df[
        "Probability (%)"
    ] = (
        probability_df["Probability"] * 100
    ).round(2)

    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )

    st.dataframe(
        probability_df[
            [
                "Cancer",
                "Probability (%)"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    ax2.bar(
        probability_df["Cancer"],
        probability_df["Probability (%)"]
    )

    ax2.set_xlabel(
        "Cancer Type"
    )

    ax2.set_ylabel(
        "Probability (%)"
    )

    ax2.set_title(
        f"Sample {selected_sample} Cancer Prediction"
    )

    ax2.set_ylim(
        0,
        100
    )

    st.pyplot(
        fig2,
        clear_figure=True
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.header(
        "🧠 SHAP Explainability"
    )

    # --------------------------------------------
    # MISCLASSIFIED SAMPLE
    # --------------------------------------------

    misclassified_file = (
        RESULTS_DIR /
        "misclassified_sample_shap.csv"
    )

    # --------------------------------------------
    # NORMAL SAMPLE SHAP FILE
    # --------------------------------------------

    individual_shap_file = (
        RESULTS_DIR /
        f"sample_{selected_sample}_SHAP.csv"
    )


    # ========================================================
    # SPECIAL CASE: EXISTING MISCLASSIFIED SAMPLE
    # ========================================================

    if (
        selected_sample == 42
        and misclassified_file.exists()
    ):

        try:

            shap_df = pd.read_csv(
                misclassified_file
            )

            st.subheader(
                "⚠️ Misclassified Sample Explanation"
            )

            st.write(
                "Sample 42 is the misclassified test sample "
                "identified during final model evaluation."
            )

            st.write(
                "**Actual:** LUAD"
            )

            st.write(
                "**Predicted:** BRCA"
            )

            if (
                "Gene" in shap_df.columns
                and "SHAP" in shap_df.columns
            ):

                shap_df[
                    "Absolute_SHAP"
                ] = shap_df["SHAP"].abs()

                top_shap = (
                    shap_df
                    .sort_values(
                        "Absolute_SHAP",
                        ascending=False
                    )
                    .head(15)
                )

                st.subheader(
                    "Top 15 Contributing Genes"
                )

                st.dataframe(
                    top_shap[
                        [
                            "Gene",
                            "SHAP",
                            "Absolute_SHAP"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                # ----------------------------------------
                # SHAP BAR CHART
                # ----------------------------------------

                plot_df = top_shap.sort_values(
                    "SHAP"
                )

                fig3, ax3 = plt.subplots(
                    figsize=(9, 6)
                )

                ax3.barh(
                    plot_df["Gene"],
                    plot_df["SHAP"]
                )

                ax3.axvline(
                    0,
                    linewidth=1
                )

                ax3.set_xlabel(
                    "SHAP Value"
                )

                ax3.set_ylabel(
                    "Gene"
                )

                ax3.set_title(
                    "Top SHAP Contributions - Sample 42"
                )

                st.pyplot(
                    fig3,
                    clear_figure=True
                )

            # Existing figure
            existing_image = (
                RESULTS_DIR /
                "SHAP_misclassified_sample_42.png"
            )

            if existing_image.exists():

                st.subheader(
                    "Existing SHAP Visualization"
                )

                st.image(
                    str(existing_image),
                    use_container_width=True
                )

        except Exception as e:

            st.warning(
                f"Could not load Sample 42 SHAP data: {e}"
            )


    # ========================================================
    # TRY SAMPLE-SPECIFIC SHAP FILE
    # ========================================================

    elif individual_shap_file.exists():

        try:

            shap_df = pd.read_csv(
                individual_shap_file
            )

            st.subheader(
                f"SHAP Explanation - Sample {selected_sample}"
            )

            if (
                "Gene" in shap_df.columns
                and "SHAP" in shap_df.columns
            ):

                shap_df[
                    "Absolute_SHAP"
                ] = shap_df["SHAP"].abs()

                top_shap = (
                    shap_df
                    .sort_values(
                        "Absolute_SHAP",
                        ascending=False
                    )
                    .head(15)
                )

                st.dataframe(
                    top_shap[
                        [
                            "Gene",
                            "SHAP",
                            "Absolute_SHAP"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                plot_df = top_shap.sort_values(
                    "SHAP"
                )

                fig4, ax4 = plt.subplots(
                    figsize=(9, 6)
                )

                ax4.barh(
                    plot_df["Gene"],
                    plot_df["SHAP"]
                )

                ax4.axvline(
                    0,
                    linewidth=1
                )

                ax4.set_xlabel(
                    "SHAP Value"
                )

                ax4.set_ylabel(
                    "Gene"
                )

                ax4.set_title(
                    f"Top SHAP Contributions - "
                    f"Sample {selected_sample}"
                )

                st.pyplot(
                    fig4,
                    clear_figure=True
                )

        except Exception as e:

            st.warning(
                f"Could not load SHAP file: {e}"
            )


    # ========================================================
    # USE EXISTING SHAP FIGURE FOR SAMPLE 42
    # ========================================================

    else:

        existing_image = (
            RESULTS_DIR /
            "SHAP_global_summary.png"
        )

        if existing_image.exists():

            st.info(
                "Individual SHAP values for this sample "
                "were not saved separately. Showing the "
                "global SHAP explanation instead."
            )

            st.image(
                str(existing_image),
                use_container_width=True
            )

        else:

            st.info(
                "SHAP explanation is not available "
                "for this sample."
            )


    # ========================================================
    # GLOBAL TOP GENES
    # ========================================================

    st.header(
        "🧬 Global SHAP Gene Importance"
    )

    global_shap_file = (
        RESULTS_DIR /
        "FINAL_GLOBAL_TOP20_SHAP_GENES.csv"
    )

    if not global_shap_file.exists():

        global_shap_file = (
            RESULTS_DIR /
            "global_shap_gene_importance.csv"
        )


    if global_shap_file.exists():

        try:

            global_df = pd.read_csv(
                global_shap_file
            )

            st.dataframe(
                global_df.head(20),
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.warning(
                f"Could not load global SHAP data: {e}"
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center">

        <b>GenoInsight</b> |
        Explainable AI for RNA-Seq Cancer Prediction

        </div>
        """,
        unsafe_allow_html=True
    )


else:

    st.info(
        "👆 Upload an RNA-Seq CSV file to begin prediction."
    )