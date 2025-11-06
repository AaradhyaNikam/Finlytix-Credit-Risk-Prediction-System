# ===========================================
# 💰 Finlytix: AI Credit Risk Intelligence
# Transparent, Explainable & Ethical AI
# ===========================================

import numpy as np
import pandas as pd
import shap
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from scipy import sparse

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Finlytix - Credit Risk AI", page_icon="💰", layout="wide")

# -----------------------------
# Helper: Extract readable feature names
# -----------------------------
def get_feature_names(pipeline):
    feature_names = []
    for name, trans, cols in pipeline.transformers_:
        if name == 'remainder':
            continue
        if hasattr(trans, 'get_feature_names_out'):
            fn = trans.get_feature_names_out(cols)
        else:
            fn = cols
        feature_names.extend(fn)
    # Clean prefixes like num__ / cat__
    return [f.split("__")[-1] for f in feature_names]

# -----------------------------
# Load model, pipeline, data
# -----------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("final_model.pkl")
    pipeline = joblib.load("final_pipeline.pkl")
    data = pd.read_csv("testing-1.csv")
    features = get_feature_names(pipeline)
    return model, pipeline, data, features

model, pipeline, test_df, feature_names = load_resources()

# -----------------------------
# Helper: Dense conversion
# -----------------------------
def ensure_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.array(X)

# -----------------------------
# Prediction
# -----------------------------
def predict_customer(cid):
    if cid not in test_df["id"].values:
        return None
    x_df = test_df.loc[test_df["id"] == cid].drop("id", axis=1)
    X_prep = pipeline.transform(x_df)
    prob_d = model.predict_proba(X_prep)[0, 1]
    prob_r = 1 - prob_d
    risk = "High" if prob_d >= 0.7 else ("Medium" if prob_d >= 0.4 else "Low")
    return prob_d, prob_r, risk, x_df, X_prep

# -----------------------------
# Explainability
# -----------------------------
def explain_customer(prepared):
    import shap
    def safe_dense(X):
        if sparse.issparse(X): return X.toarray()
        return np.array(X)

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(prepared)
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        shap_vals = safe_dense(shap_vals)
    except Exception:
        print("TreeExplainer failed — fallback to KernelExplainer.")
        background = shap.sample(prepared, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_vals = explainer.shap_values(prepared, nsamples=100)
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        shap_vals = safe_dense(shap_vals)

    shap_vals = np.nan_to_num(shap_vals, nan=0.0, posinf=0.0, neginf=0.0)
    shap_vals = shap_vals.flatten().astype(float, copy=False)
    if shap_vals.size == 0:
        return np.zeros(len(feature_names)), [], []

    shap_vals *= 1000  # scale for better visibility
    order = np.argsort(np.abs(shap_vals))[::-1][:10]
    top = [(feature_names[i], float(shap_vals[i])) for i in order if np.isfinite(shap_vals[i])]

    pos, neg = [], []
    for f, v in top:
        if v > 0.001:
            pos.append((f, v))
        elif v < -0.001:
            neg.append((f, v))

    return shap_vals, pos, neg

# -----------------------------
# Utility: Format SHAP factors
# -----------------------------
def pretty_factors(factors):
    lines = []
    for f, v in factors[:5]:
        arrow = "↑" if v > 0 else "↓"
        lines.append(f"- {f}: {arrow} ({v:+.3f})")
    return "\n".join(lines) if lines else "No strong factors detected."

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏦 Finlytix Navigation")
mode = st.sidebar.radio("Select Mode", ["📊 Dashboard", "💬 Chatbot"])

# ============================================================
# 📊 DASHBOARD MODE
# ============================================================
if mode == "📊 Dashboard":
    st.title("💰 Finlytix: AI Credit Risk Intelligence")
    st.caption("Transparent, Explainable & Ethical AI for Smart Financial Decisions")

    cid = st.text_input("Enter Customer id:")
    if cid:
        try:
            cid = int(cid)
            result = predict_customer(cid)
            if result:
                prob_d, prob_r, risk, x_df, X_prep = result
                st.subheader(f"Prediction for Customer id {cid}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Default Probability", f"{prob_d:.2%}")
                col2.metric("Repayment Probability", f"{prob_r:.2%}")
                col3.metric("Risk Level", risk)

                shap_vals, pos, neg = explain_customer(X_prep)

                st.markdown("### 🔍 Increasing & Decreasing Risk Factors")
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**⬆️ Increasing Risk Factors (higher default chance)**")
                    st.markdown(pretty_factors(pos))
                with colB:
                    st.markdown("**⬇️ Decreasing Risk Factors (safer customer)**")
                    st.markdown(pretty_factors(neg))

                # Visual SHAP bar chart
                st.markdown("### 📊 Visual SHAP Impact (Top 10 Features)")
                plt.figure(figsize=(8,4))
                labels = [f for f,_ in pos+neg]
                values = [v for _,v in pos+neg]
                colors = ['#E53935' if v>0 else '#43A047' for v in values]
                plt.barh(labels[::-1], values[::-1], color=colors[::-1])
                plt.xlabel("SHAP Value (Impact on Risk)")
                st.pyplot(plt.gcf())
                plt.clf()

                # Global SHAP Summary Plot
                st.markdown("---")
                st.markdown("### 🌍 Global SHAP Summary (Feature Importance)")
                try:
                    sample = shap.sample(pipeline.transform(test_df.drop(columns=['id'])), 400)
                    expl = shap.TreeExplainer(model)
                    global_vals = expl.shap_values(sample)
                    if isinstance(global_vals, list): global_vals = global_vals[1]
                    shap.summary_plot(global_vals, features=sample, feature_names=feature_names, show=False, plot_size=(8,4))
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception:
                    st.info("Global SHAP summary unavailable in this environment.")

                # Fairness Analysis
                st.markdown("---")
                st.markdown("### ⚖️ Fairness Analysis: Influence of MonthlyIncome")
                try:
                    mean_abs = np.abs(global_vals).mean(axis=0)
                    total = mean_abs.sum()
                    idx = [i for i,f in enumerate(feature_names) if "MonthlyIncome" in f]
                    income_share = mean_abs[idx].sum()/total*100 if idx else 0
                    st.write(f"💡 MonthlyIncome contributes **{income_share:.2f}%** of total model reasoning.")
                    if income_share > 25:
                        st.warning("Model relies heavily on income — review for bias.")
                    else:
                        st.success("Income influence within fair and ethical range.")
                except Exception:
                    st.info("Fairness metrics unavailable for this run.")

                with st.expander("📋 View Customer Data"):
                    st.dataframe(x_df)
            else:
                st.error("id not found in dataset.")
        except ValueError:
            st.error("Please enter a valid numeric id.")

# ============================================================
# 💬 CHATBOT MODE
# ============================================================
elif mode == "💬 Chatbot":
    st.header("💬 Finlytix Explainable Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_query = st.chat_input("Ask: check id <number>")
    if user_query:
        st.session_state.history.append(("user", user_query))
        import re
        match = re.search(r"(\d+)", user_query)
        if not match:
            st.session_state.history.append(("assistant", "Please include a numeric id (e.g., check id 1034)."))
        else:
            cid = int(match.group(1))
            result = predict_customer(cid)
            if not result:
                st.session_state.history.append(("assistant", f"❌ id {cid} not found."))
            else:
                prob_d, prob_r, risk, x_df, X_prep = result
                shap_vals, pos, neg = explain_customer(X_prep)
                msg = (f"**Customer id {cid}** — Risk: **{risk}**\n\n"
                       f"- Default Probability: **{prob_d:.2%}**\n"
                       f"- Repayment Probability: **{prob_r:.2%}**\n\n"
                       "### 🔍 Increasing Risk Factors\n" + pretty_factors(pos) +
                       "\n\n### 🔍 Decreasing Risk Factors\n" + pretty_factors(neg))
                st.session_state.history.append(("assistant", msg))

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

