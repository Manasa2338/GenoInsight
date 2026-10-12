
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GenoInsight",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results_vae_dnn"
EXPLAIN_DIR = MODEL_DIR / "explainability"
DATA_DIR = BASE_DIR / "data" / "processed"

VAE_PATH = MODEL_DIR / "vae_model.pt"
DNN_PATH = MODEL_DIR / "dnn_model.pt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

FEATURES_PATH = DATA_DIR / "X_train_selected.csv"

DEVICE = torch.device("cpu")
LATENT_DIM = 64

# ============================================================
# MODEL ARCHITECTURES
# Must match train_vae_dnn.py exactly
# ============================================================

class VAE(nn.Module):
    def __init__(self, input_dim=5000, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return mu, logvar


class DNN(nn.Module):
    def __init__(self, input_dim=64, num_classes=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD TRAINED OBJECTS
# ============================================================

@st.cache_resource
def load_objects():
    import joblib

    required = [
        VAE_PATH, DNN_PATH, SCALER_PATH,
        ENCODER_PATH, FEATURES_PATH
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n" + "\n".join(missing)
        )

    feature_df = pd.read_csv(FEATURES_PATH)
    feature_df = feature_df.loc[
        :, ~feature_df.columns.astype(str).str.startswith("Unnamed:")
    ]

    feature_names = feature_df.columns.astype(str).tolist()

    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    vae = VAE(input_dim=len(feature_names), latent_dim=LATENT_DIM)
    dnn = DNN(input_dim=LATENT_DIM, num_classes=len(label_encoder.classes_))

    vae.load_state_dict(
        torch.load(VAE_PATH, map_location=DEVICE, weights_only=True)
    )
    dnn.load_state_dict(
        torch.load(DNN_PATH, map_location=DEVICE, weights_only=True)
    )

    vae.to(DEVICE).eval()
    dnn.to(DEVICE).eval()

    return vae, dnn, scaler, label_encoder, feature_names


# ============================================================
# HEADER
# ============================================================

st.title("🧬 GenoInsight")
st.subheader("Explainable AI for RNA-Seq Cancer Prediction")

st.write(
    "GenoInsight uses a Variational Autoencoder (VAE) and "
    "Deep Neural Network (DNN) to classify RNA-Seq expression "
    "profiles into five cancer classes, with Integrated Gradients "
    "for model-level explanations."
)

try:
    vae, dnn, scaler, label_encoder, selected_genes = load_objects()
    classes = [str(c) for c in label_encoder.classes_]
except Exception as e:
    st.error(f"Could not load the VAE + DNN model: {e}")
    st.stop()

# ============================================================
# MODEL INFORMATION
# ============================================================

metrics_path = MODEL_DIR / "test_metrics.csv"

st.header("📊 Model Information")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Selected Features", f"{len(selected_genes):,}")
c2.metric("Cancer Classes", len(classes))
c3.metric("Latent Dimensions", LATENT_DIM)

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path).iloc[0]
    c4.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    st.metric("Test Macro F1", f"{metrics['macro_f1']:.4f}")
else:
    c4.metric("Test Accuracy", "Not available")

st.info("Active model: Variational Autoencoder (VAE) + Deep Neural Network (DNN)")

# ============================================================
# FEATURE ATTRIBUTION
# Integrated Gradients through VAE latent means and DNN logits
# ============================================================

def integrated_gradients(x, target_class, steps=32):
    """
    x: one already-scaled sample with shape (1, 5000).
    Baseline: zero in standardized feature space.
    Attribution target: selected class logit.
    """
    baseline = torch.zeros_like(x)
    difference = x - baseline
    total_gradients = torch.zeros_like(x)

    for alpha in torch.linspace(0, 1, steps, device=DEVICE):
        interpolated = (
            baseline + alpha * difference
        ).detach().requires_grad_(True)

        mu, _ = vae.encode(interpolated)
        logits = dnn(mu)

        score = logits[0, target_class]
        gradients = torch.autograd.grad(
            score, interpolated, retain_graph=False
        )[0]

        total_gradients += gradients.detach()

    average_gradients = total_gradients / steps
    attributions = difference * average_gradients

    return attributions.detach().cpu().numpy().flatten()


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📁 Upload RNA-Seq Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV containing RNA-Seq expression values",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin prediction.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)

    expression_df = raw_df.drop(
        columns=[
            c for c in raw_df.columns
            if str(c).startswith("Unnamed:")
        ],
        errors="ignore"
    )

    # Keep only the selected features, in training order.
    missing = [
        gene for gene in selected_genes
        if gene not in expression_df.columns
    ]

    if missing:
        st.error(
            f"{len(missing)} required selected features are missing. "
            "Upload data containing the original gene_XXXXX columns "
            "or the exact 5,000 selected features used during training."
        )
        st.write("Examples of missing features:", missing[:20])
        st.stop()

    X_selected = expression_df[selected_genes].apply(
        pd.to_numeric, errors="coerce"
    )

    if X_selected.isna().any().any():
        st.error(
            "Selected features contain missing or non-numeric values. "
            "Please clean the input CSV and upload it again."
        )
        st.stop()

    if not np.isfinite(X_selected.to_numpy(dtype=np.float64)).all():
        st.error("Selected features contain infinite values.")
        st.stop()

    X_scaled = scaler.transform(X_selected).astype(np.float32)

    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        mu, _ = vae.encode(x_tensor)
        logits = dnn(mu)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_ids = probs.argmax(axis=1)

    predictions = label_encoder.inverse_transform(pred_ids)

except Exception as e:
    st.error(f"Error processing uploaded dataset: {e}")
    st.stop()

# ============================================================
# DATASET SUMMARY
# ============================================================

st.header("📊 Dataset Information")

c1, c2, c3 = st.columns(3)
c1.metric("Samples", X_selected.shape[0])
c2.metric("Uploaded Columns", raw_df.shape[1])
c3.metric("Selected Features Used", len(selected_genes))

with st.expander("Preview uploaded data"):
    st.dataframe(raw_df.head(10), use_container_width=True)

# ============================================================
# PREDICTIONS
# ============================================================

st.header("🔬 Cancer Prediction")
st.success(f"Predictions completed for {len(predictions)} samples.")

result_df = pd.DataFrame({
    "Sample": [f"Sample {i}" for i in range(len(predictions))],
    "Predicted Cancer": predictions
})

for i, cancer in enumerate(classes):
    result_df[f"Probability_{cancer} (%)"] = np.round(probs[:, i] * 100, 2)

st.dataframe(result_df, use_container_width=True, height=300)

st.download_button(
    "Download prediction results",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="genoinsight_predictions.csv",
    mime="text/csv"
)

# ============================================================
# PREDICTION SUMMARY
# ============================================================

st.header("📈 Predicted Cancer Distribution")

counts = pd.Series(predictions).value_counts().reindex(classes, fill_value=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(counts.index, counts.values)
ax.set_xlabel("Cancer Class")
ax.set_ylabel("Number of Samples")
ax.set_title("Predicted Cancer Distribution")
st.pyplot(fig, clear_figure=True)
plt.close(fig)

# ============================================================
# INDIVIDUAL SAMPLE EXPLANATION
# ============================================================

st.header("🧠 Individual Sample Explanation")

sample_index = st.selectbox(
    "Select a sample",
    range(len(predictions)),
    format_func=lambda i: f"Sample {i}"
)

predicted_class = str(predictions[sample_index])
predicted_id = int(pred_ids[sample_index])

st.subheader(f"Sample {sample_index}")
c1, c2 = st.columns(2)
c1.metric("Predicted Cancer Class", predicted_class)
c2.metric(
    "Top Class Probability",
    f"{probs[sample_index, predicted_id] * 100:.2f}%"
)

st.caption(
    "These are model probabilities, not calibrated clinical probabilities "
    "or a medical diagnosis."
)

st.subheader("Class Probabilities")

probability_df = pd.DataFrame({
    "Cancer Class": classes,
    "Probability (%)": np.round(probs[sample_index] * 100, 2)
}).sort_values("Probability (%)", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    probability_df["Cancer Class"],
    probability_df["Probability (%)"]
)
ax.set_ylabel("Probability (%)")
ax.set_ylim(0, 100)
ax.set_title(f"Class Probabilities — Sample {sample_index}")
st.pyplot(fig, clear_figure=True)
plt.close(fig)

st.dataframe(probability_df, use_container_width=True, hide_index=True)

# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

st.subheader("Integrated Gradients: Top Influential Features")

st.write(
    "Attributions explain how input features affect the selected "
    "class logit relative to a zero baseline in standardized feature space."
)

if st.button("Generate explanation for this sample"):
    with st.spinner("Calculating Integrated Gradients..."):
        x_one = torch.tensor(
            X_scaled[sample_index:sample_index + 1],
            dtype=torch.float32,
            device=DEVICE
        )

        attribution_values = integrated_gradients(
            x_one, predicted_id, steps=32
        )

        attribution_df = pd.DataFrame({
            "Gene Feature": selected_genes,
            "Attribution": attribution_values
        })

        attribution_df["Absolute Attribution"] = (
            attribution_df["Attribution"].abs()
        )

        top_df = attribution_df.sort_values(
            "Absolute Attribution", ascending=False
        ).head(10)

        st.dataframe(
            top_df[["Gene Feature", "Attribution"]],
            use_container_width=True,
            hide_index=True
        )

        plot_df = top_df.sort_values("Attribution")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            plot_df["Gene Feature"],
            plot_df["Attribution"]
        )
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Integrated Gradients Attribution")
        ax.set_ylabel("Gene Feature")
        ax.set_title(
            f"Top Feature Attributions — Sample {sample_index} "
            f"({predicted_class})"
        )
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        csv_data = top_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download sample explanation",
            data=csv_data,
            file_name=f"sample_{sample_index}_integrated_gradients.csv",
            mime="text/csv"
        )

# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

st.header("🌐 Global Feature Importance")

global_file = EXPLAIN_DIR / "global_gene_importance.csv"

if global_file.exists():
    global_df = pd.read_csv(global_file)
    st.dataframe(global_df.head(20), use_container_width=True, hide_index=True)

    gene_col = next(
        (c for c in ["gene", "Gene", "feature", "Feature"]
         if c in global_df.columns),
        None
    )
    score_col = next(
        (c for c in ["mean_absolute_attribution", "mean_abs_attribution",
                     "mean_absolute_ig", "importance"]
         if c in global_df.columns),
        None
    )

    if gene_col and score_col:
        plot_df = global_df.head(15).sort_values(score_col)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(plot_df[gene_col], plot_df[score_col])
        ax.set_xlabel("Mean Absolute Attribution")
        ax.set_title("Global Integrated Gradients Feature Importance")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
else:
    st.info(
        "Global attribution file not found. Individual explanations "
        "can still be generated above."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center'><b>GenoInsight</b> | "
    "VAE + DNN with Integrated Gradients</div>",
    unsafe_allow_html=True
)
