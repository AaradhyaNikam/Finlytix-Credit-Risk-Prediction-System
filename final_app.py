# ===========================================
# Finlytix Unified App — Explainability + Fairness Ready
# ===========================================

import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from scipy import sparse

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------
st.set_page_config(page_title="Finlytix - AI Credit Risk Intelligence",
                   page_icon="💰", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------
# Load model + data
# ---------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("final_model.pkl")
    pipeline = joblib.load("final_pipeline.pkl")
    test_df = pd.read_csv("testing-1.csv")
    try:
        features = pipeline.get_feature_names_out()
    except Exception:
        features = np.array([f"f_{i}" for i in range(pipeline.transform(test_df.drop(columns=['id'])).shape[1])])
    return model, pipeline, test_df, features

model, pipeline, test_df, feature_names = load_resources()

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def ensure_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.array(X)

def predict_customer(cid: int):
    if cid not in test_df["id"].values:
        return None
    x_df = test_df.loc[test_df["id"] == cid].drop("id", axis=1)
    X_prepared = pipeline.transform(x_df)
    prob_d = model.predict_proba(X_prepared)[0, 1]
    prob_r = 1 - prob_d
    risk = "High" if prob_d >= 0.7 else ("Medium" if prob_d >= 0.4 else "Low")
    return prob_d, prob_r, risk, x_df, X_prepared

# ---------------------------------------------------
# Explainability (Tree→Kernel fallback)
# ---------------------------------------------------
# ---------------------------------------------------
# Safe SHAP + LIME Explainability (Streamlit Cloud)
# ---------------------------------------------------
def explain_customer(prepared):
    import shap
    import numpy as np

    def safe_dense(X):
        from scipy import sparse
        if sparse.issparse(X):
            return X.toarray()
        return np.array(X)

    try:
        # Try fast TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(prepared)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = safe_dense(shap_vals)[0]
    except Exception:
        # Safe fallback for Streamlit Cloud
        print("TreeExplainer failed — fallback to KernelExplainer.")
        bg = shap.sample(prepared, 50)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_vals = explainer.shap_values(prepared, nsamples=100)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = safe_dense(shap_vals)[0]

    # --- Clean and validate SHAP values ---
    shap_vals = np.nan_to_num(shap_vals, nan=0.0)  # replace NaN/inf with 0
    shap_vals = shap_vals.astype(float, copy=False)  # force numeric dtype

    # Guard for zero-length outputs
    if shap_vals.shape[0] == 0:
        return shap_vals, [], []

    # Compute top absolute impacts
    order = np.argsort(np.abs(shap_vals))[::-1]
    top = [(feature_names[i], shap_vals[i]) for i in order[:10]
           if np.isfinite(shap_vals[i])]

    # Safely separate positive/negative with numeric filter
    pos, neg = [], []
    for f, v in top:
        if isinstance(v, (int, float, np.floating)):
            if v > 0:
                pos.append((f, float(v)))
            elif v < 0:
                neg.append((f, float(v)))

    return shap_vals, pos, neg


def pretty_factors(factors):
    lines = []
    for f, v in factors[:5]:
        arrow = "↑" if v > 0 else "↓"
        lines.append(f"- {f}: {arrow} ({v:+.4f})")
    return "\n".join(lines) if lines else "No strong factors detected."

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("🏦 Finlytix Navigation")
mode = st.sidebar.radio("Select Mode", ["📊 Dashboard", "💬 Chatbot"])

# ===================================================
# DASHBOARD MODE
# ===================================================
if mode == "📊 Dashboard":
    st.title("💰 Finlytix: AI Credit Risk Intelligence")
    st.caption("Transparent, Explainable & Ethical AI for Smart Financial Decisions")

    cid = st.text_input("Enter Customer id:")
    if cid:
        try:
            cid = int(cid)
            result = predict_customer(cid)
            if result:
                prob_d, prob_r, risk, x_df, X_prepared = result
                st.subheader(f"Prediction for Customer id {cid}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Default Probability", f"{prob_d:.2%}")
                col2.metric("Repayment Probability", f"{prob_r:.2%}")
                col3.metric("Risk Level", risk)

                shap_vals, pos, neg = explain_customer(X_prepared)

                st.markdown("### 🔍 Increasing & Decreasing Risk Factors (SHAP)")
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**⬆️ Increasing Risk Factors**")
                    st.markdown(pretty_factors(pos))
                with colB:
                    st.markdown("**⬇️ Decreasing Risk Factors**")
                    st.markdown(pretty_factors(neg))

                # --- Global SHAP Summary Plot ---
                st.markdown("---")
                st.markdown("### 🌍 Global SHAP Summary (Feature Importance)")
                sample = shap.sample(pipeline.transform(test_df.drop(columns=['id'])), 500)
                try:
                    global_explainer = shap.TreeExplainer(model)
                    global_vals = global_explainer.shap_values(sample)
                    if isinstance(global_vals, list): global_vals = global_vals[1]
                    shap.summary_plot(global_vals, features=sample,
                                      feature_names=feature_names, show=False, plot_size=(8,4))
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception:
                    st.info("Global SHAP Summary unavailable for this environment.")

                # --- Fairness Analysis ---
                st.markdown("---")
                st.markdown("### ⚖️ Fairness Analysis: Influence of MonthlyIncome")
                try:
                    mean_abs = np.abs(global_vals).mean(axis=0)
                    total = mean_abs.sum()
                    income_idx = [i for i, f in enumerate(feature_names)
                                  if "MonthlyIncome" in f]
                    income_share = mean_abs[income_idx].sum() / total * 100 if income_idx else 0
                    st.write(f"💡 MonthlyIncome contributes **{income_share:.2f}%** "
                             "to the model's overall decision reasoning.")
                    if income_share > 25:
                        st.warning("Model relies heavily on income — review for fairness bias.")
                    else:
                        st.success("Income influence within fair, ethical limits.")
                except Exception:
                    st.info("Fairness metric not available in this session.")

                with st.expander("View Customer Data"):
                    st.dataframe(x_df)
            else:
                st.error("id not found in database.")
        except ValueError:
            st.error("Please enter a valid numeric id.")

# ===================================================
# CHATBOT MODE
# ===================================================
elif mode == "💬 Chatbot":
    st.header("💬 Finlytix Explainable Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_query = st.chat_input("Ask me: check id <number>")

    if user_query:
        st.session_state.history.append(("user", user_query))
        import re
        m = re.search(r"(\d+)", user_query)
        if not m:
            st.session_state.history.append(("assistant", "Please include a numeric id (e.g., check id 1034)."))
        else:
            cid = int(m.group(1))
            result = predict_customer(cid)
            if not result:
                st.session_state.history.append(("assistant", f"❌ id {cid} not found in database."))
            else:
                prob_d, prob_r, risk, x_df, X_prepared = result
                shap_vals, pos, neg = explain_customer(X_prepared)
                reply = (f"**Customer id {cid}** — Risk: **{risk}**\n\n"
                         f"- Default Probability: **{prob_d:.2%}**\n"
                         f"- Repayment Probability: **{prob_r:.2%}**\n\n"
                         "### 🔍 Increasing Risk Factors\n" + pretty_factors(pos) +
                         "\n\n### 🔍 Decreasing Risk Factors\n" + pretty_factors(neg))
                st.session_state.history.append(("assistant", reply))

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

