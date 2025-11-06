# ===========================================
# Finlytix Chatbot – Risk + Explainability (SHAP + LIME)
# ===========================================

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.utils import Bunch
from scipy import sparse

# Explainability
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Finlytix Chatbot", page_icon="💬", layout="centered")
st.title("💬 Finlytix Chatbot: Credit Risk & Explanations")
st.caption("Ask me about a customer by Id. I’ll predict risk and explain *why* using SHAP & LIME.")

# ---------- Paths ----------
MODEL_FILE = "final_model.pkl"
PIPELINE_FILE = "final_pipeline.pkl"
TEST_FILE = "testing-1.csv"

# ---------- Cache resources ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Feature names from pipeline
    try:
        feat_names = pipeline.get_feature_names_out()
    except:
        # Fallback: create generic names if not available
        feat_names = np.array([f"f_{i}" for i in range(pipeline.transform(test_df.drop(columns=['Id'])).shape[1])])

    return Bunch(model=model, pipeline=pipeline, test_df=test_df, feature_names=feat_names)

art = load_artifacts()

# ---------- Utilities ----------
def _ensure_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.array(X)

def _predict_and_explain_by_id(customer_id: int, high_risk_threshold: float = 0.70):
    # Validate id
    if customer_id not in art.test_df["id"].values:
        return {"error": "Invalid Customer Id. Please choose an Id present in testing-1.csv."}

    row_df = art.test_df.loc[art.test_df["id"] == customer_id]
    features_df = row_df.drop(columns=["id"])
    X_trans = art.pipeline.transform(features_df)
    X_dense = _ensure_dense(X_trans)

    # Prediction
    prob_default = float(art.model.predict_proba(X_dense)[0, 1])
    prob_repay = 1.0 - prob_default
    risk_label = "High" if prob_default >= high_risk_threshold else ("Medium" if prob_default >= 0.40 else "Low")

    # ---------- SHAP ----------
    # TreeExplainer works great for XGBoost
    explainer = shap.TreeExplainer(art.model)
    shap_values = explainer.shap_values(X_trans)  # works with sparse
    # Convert to dense for sorting & display
    shap_vals_dense = _ensure_dense(shap_values)[0]  # 1D for the instance

    # Top + and - drivers
    fn = art.feature_names
    order = np.argsort(np.abs(shap_vals_dense))[::-1]
    top_k = 8 if len(order) >= 8 else len(order)
    top_idx = order[:top_k]

    contributions = [
        (fn[i], shap_vals_dense[i])
        for i in top_idx
    ]
    positive = [(f, v) for f, v in contributions if v > 0]
    negative = [(f, v) for f, v in contributions if v < 0]

    # ---------- LIME ----------
    # Build explainer over the whole transformed feature space
    # We need a training matrix for LIME; sample a chunk to keep it lightweight
    sample_df = art.test_df.sample(min(len(art.test_df), 500), random_state=42).drop(columns=["id"])
    X_sample = _ensure_dense(art.pipeline.transform(sample_df))

    lime_explainer = LimeTabularExplainer(
        training_data=X_sample,
        feature_names=art.feature_names,
        class_names=["Repay", "Default"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    def predict_fn(xx):
        # LIME passes raw numpy; ensure proper dtype/shape
        return art.model.predict_proba(xx)

    lime_exp = lime_explainer.explain_instance(
        X_dense[0],
        predict_fn,
        num_features=min(10, len(art.feature_names))
    )
    lime_list = lime_exp.as_list()

    return {
        "row_df": row_df,
        "features_df": features_df,
        "prob_default": prob_default,
        "prob_repay": prob_repay,
        "risk_label": risk_label,
        "shap": {
            "values": shap_vals_dense,
            "top_contributions": contributions,
            "positive": positive,
            "negative": negative,
        },
        "lime": lime_list
    }

def _pretty_factors(factors, max_items=5):
    lines = []
    for name, val in factors[:max_items]:
        arrow = "↑" if val > 0 else "↓"
        lines.append(f"- {name}: {arrow} impact ({val:+.4f})")
    return "\n".join(lines) if lines else "• (No strong factors detected)"

# ---------- Chat UI ----------
if "history" not in st.session_state:
    st.session_state.history = []

with st.chat_message("assistant"):
    st.markdown("Hi! Give me a **Customer Id** from your database (`testing-1.csv`). I’ll predict risk and explain why using SHAP & LIME.")

user_id_text = st.chat_input("Type: check id <number>  e.g.,  check id 1234")

if user_id_text:
    st.session_state.history.append(("user", user_id_text))

# Render history (including new message)
for role, text in st.session_state.history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(text)

# On new user input, parse and respond
if user_id_text:
    # Parse something like "check id 1234"
    import re
    m = re.search(r"(\d+)", user_id_text)
    if not m:
        reply = "Please include a numeric Id. For example: `check id 10542`."
        st.session_state.history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        cid = int(m.group(1))
        result = _predict_and_explain_by_id(cid)

        if "error" in result:
            reply = f"❌ {result['error']}"
            st.session_state.history.append(("assistant", reply))
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            prob_d = result["prob_default"]
            prob_r = result["prob_repay"]
            label = result["risk_label"]

            # Compose a friendly explanation
            header = f"**Customer Id {cid}** — Risk: **{label}**\n\n" \
                     f"- Probability of **Default**: **{prob_d:.2%}**\n" \
                     f"- Probability of **Repayment**: **{prob_r:.2%}**"

            # SHAP: top drivers
            pos_txt = _pretty_factors(result["shap"]["positive"])
            neg_txt = _pretty_factors(result["shap"]["negative"])

            expl_txt = (
                "### 🔍 Why?\n"
                "**Top factors pushing risk *up*** (SHAP):\n"
                f"{pos_txt}\n\n"
                "**Top factors pushing risk *down*** (SHAP):\n"
                f"{neg_txt}\n"
            )

            # LIME summary
            # LIME tuples like [('feature_name <= 0.12', +0.14), ...]
            lime_lines = []
            for feat, val in result["lime"][:6]:
                arrow = "↑" if val > 0 else "↓"
                lime_lines.append(f"- {feat}: {arrow} ({val:+.4f})")
            lime_txt = "### 🟡 LIME perspective (local rule-of-thumb)\n" + "\n".join(lime_lines)

            full_reply = header + "\n\n" + expl_txt + "\n" + lime_txt

            st.session_state.history.append(("assistant", full_reply))
            with st.chat_message("assistant"):
                st.markdown(full_reply)

            # Optional: show the raw row
            with st.expander("View customer raw features"):
                st.dataframe(result["row_df"].set_index("id"))

            # Optional: SHAP bar plot (top absolute)
            with st.expander("View SHAP bar chart (top absolute impacts)"):
                shap_vals = result["shap"]["values"]
                names = art.feature_names
                order = np.argsort(np.abs(shap_vals))[::-1][:15]
                plt.figure(figsize=(7, 4))
                plt.barh([names[i] for i in order][::-1], np.abs(shap_vals[order])[::-1])
                plt.xlabel("Absolute SHAP impact")
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.clf()
