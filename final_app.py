# ===========================================
# Finlytix Unified App — Dashboard + Chatbot (Themed)
# ===========================================

import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from scipy import sparse

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Finlytix - AI Credit Risk System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Inject custom CSS theme
# -------------------------------
st.markdown("""
<style>
/* Global App Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #0B1E3F, #13294B);
    color: #E8EAF6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #09162C;
    color: #E8EAF6;
}

/* Metrics & Cards */
[data-testid="stMetricValue"] {
    color: #FDD835 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #101C33 !important;
    border: 1px solid #2F3C5A;
    border-radius: 10px;
}

/* Headers */
h1, h2, h3 {
    color: #FFD54F;
}

/* Buttons */
div.stButton > button {
    background-color: #FFD54F;
    color: #0B1E3F;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #FFEE58;
    color: #13294B;
}

/* Text inputs */
input, textarea {
    border-radius: 6px !important;
    background-color: #E8EAF6 !important;
    color: #0B1E3F !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header section
# -------------------------------
st.markdown("""
<div style="text-align:center; padding:1rem 0; background-color:#09162C; border-radius:10px;">
    <h1 style="color:#FFD54F; font-family: 'Trebuchet MS'; font-weight:800;">
        💰 Finlytix: AI Credit Risk Intelligence
    </h1>
    <h4 style="color:#C5CAE9;">Transparent, Explainable & Ethical AI for Smart Financial Decisions</h4>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load model, pipeline, and data
# -------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("final_model.pkl")
    pipeline = joblib.load("final_pipeline.pkl")
    data = pd.read_csv("testing-1.csv")
    try:
        feature_names = pipeline.get_feature_names_out()
    except:
        feature_names = np.array([f"feature_{i}" for i in range(pipeline.transform(data.drop(columns=['id'])).shape[1])])
    return model, pipeline, data, feature_names

model, pipeline, test_data, feature_names = load_resources()

# -------------------------------
# Helper functions
# -------------------------------
def ensure_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.array(X)

def predict_customer(cid):
    if cid not in test_data["id"].values:
        return None
    person = test_data.loc[test_data["id"] == cid].drop("id", axis=1)
    prepared = pipeline.transform(person)
    prob_default = model.predict_proba(prepared)[0, 1]
    prob_repay = 1 - prob_default
    risk = "High" if prob_default >= 0.7 else ("Medium" if prob_default >= 0.4 else "Low")
    return prob_default, prob_repay, risk, person, prepared

# ---------------------------------------------
# Safe SHAP + LIME Explanation (Cloud-Compatible)
# ---------------------------------------------
# ---------------------------------------------------------
# Finlytix Safe SHAP + LIME Explainability (Cloud Compatible)
# ---------------------------------------------------------
def explain_customer(prepared):
    import shap
    import numpy as np

    def safe_dense(X):
        from scipy import sparse
        if sparse.issparse(X):
            return X.toarray()
        return np.array(X)

    # Try TreeExplainer first (fast)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(prepared)
        # Handle multiclass outputs (XGBClassifier returns 2-class list)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = safe_dense(shap_vals)[0]

    except Exception as e:
        # Fallback to KernelExplainer (safe for Streamlit Cloud)
        st.warning("⚠️ TreeExplainer failed, switching to KernelExplainer for compatibility.")
        background = shap.sample(prepared, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_vals = explainer.shap_values(prepared, nsamples=100)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = safe_dense(shap_vals)[0]

    # --- Handle invalid / empty shap values safely ---
    shap_vals = np.nan_to_num(shap_vals, nan=0.0)

    # Compute top features safely
    order = np.argsort(np.abs(shap_vals))[::-1]
    if len(order) == 0:
        return shap_vals, [], []

    top = [(feature_names[i], shap_vals[i]) for i in order[:10]]

    # Filter with numeric safety
    pos = [(f, v) for f, v in top if isinstance(v, (int, float)) and v > 0]
    neg = [(f, v) for f, v in top if isinstance(v, (int, float)) and v < 0]

    return shap_vals, pos, neg



def pretty_factors(factors):
    txt = []
    for f, v in factors[:5]:
        arrow = "↑" if v > 0 else "↓"
        txt.append(f"- {f}: {arrow} ({v:+.4f})")
    return "\n".join(txt) if txt else "No strong factors found."

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🏦 Finlytix Navigation")
mode = st.sidebar.radio("Select Mode", ["📊 Dashboard", "💬 Chatbot"])

# =========================================================
# 📊 DASHBOARD MODE
# =========================================================
if mode == "📊 Dashboard":
    st.header("📊 Credit Risk Prediction Dashboard")

    cid = st.text_input("Enter Customer id:")
    if cid:
        try:
            cid = int(cid)
            result = predict_customer(cid)
            if result:
                prob_default, prob_repay, risk, person, prepared = result
                col1, col2, col3 = st.columns(3)
                col1.metric("Probability of Default", f"{prob_default:.2%}")
                col2.metric("Probability of Repayment", f"{prob_repay:.2%}")
                col3.metric("Risk Level", risk)

                if risk == "High":
                    st.error("🔴 High Risk: Customer likely to default.")
                elif risk == "Medium":
                    st.warning("🟠 Medium Risk: Needs close monitoring.")
                else:
                    st.success("🟢 Low Risk: Customer likely to repay.")

                shap_vals, pos, neg = explain_customer(prepared)

                with st.expander("🔍 SHAP Insights"):
                    st.markdown("**Top factors increasing risk:**")
                    st.markdown(pretty_factors(pos))
                    st.markdown("**Top factors decreasing risk:**")
                    st.markdown(pretty_factors(neg))

                    order = np.argsort(np.abs(shap_vals))[::-1][:10]
                    plt.figure(figsize=(7, 4))
                    plt.barh([feature_names[i] for i in order][::-1], np.abs(shap_vals[order])[::-1])
                    plt.xlabel("Absolute SHAP Impact")
                    plt.tight_layout()
                    st.pyplot(plt.gcf())

                with st.expander("Customer Data"):
                    st.dataframe(person)
            else:
                st.error("id not found in dataset.")
        except ValueError:
            st.error("Please enter a valid numeric id.")

# =========================================================
# 💬 CHATBOT MODE
# =========================================================
elif mode == "💬 Chatbot":
    st.header("💬 Finlytix Explainable Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_query = st.chat_input("Ask: check id <number>")

    if user_query:
        st.session_state.history.append(("user", user_query))
        import re
        m = re.search(r"(\d+)", user_query)
        if not m:
            st.session_state.history.append(("assistant", "Please enter a valid numeric id."))
        else:
            cid = int(m.group(1))
            result = predict_customer(cid)
            if not result:
                st.session_state.history.append(("assistant", f"❌ id {cid} not found in database."))
            else:
                prob_d, prob_r, risk, person, prepared = result
                shap_vals, pos, neg = explain_customer(prepared)

                header = f"**Customer id {cid}**\n- Risk Level: **{risk}**\n- Default Probability: **{prob_d:.2%}**\n- Repayment Probability: **{prob_r:.2%}**"
                reason = (
                    "### 🔍 Top Risk Factors\n"
                    "**Increase risk:**\n" + pretty_factors(pos) +
                    "\n\n**Decrease risk:**\n" + pretty_factors(neg)
                )

                st.session_state.history.append(("assistant", header + "\n\n" + reason))

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)
