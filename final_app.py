# ===========================================
# 💰 Finlytix: AI Credit Risk Intelligence (EY-Ready, Error-Proof)
# ===========================================

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from scipy import sparse
import shap
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Finlytix - Credit Risk AI", page_icon="💰", layout="wide")

# -----------------------------
# Safe helpers
# -----------------------------
def to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.array(X)

def safe_feature_names_from_pipeline(pipeline, fallback_cols):
    names = []
    try:
        if hasattr(pipeline, "get_feature_names_out"):
            names = list(pipeline.get_feature_names_out())
        elif hasattr(pipeline, "transformers_"):
            for name, trans, cols in pipeline.transformers_:
                if name == "remainder":
                    continue
                try:
                    if hasattr(trans, "get_feature_names_out"):
                        fn = list(trans.get_feature_names_out(cols))
                    elif hasattr(trans, "named_steps"):
                        last = list(trans.named_steps.values())[-1]
                        if hasattr(last, "get_feature_names_out"):
                            fn = list(last.get_feature_names_out(cols))
                        else:
                            fn = list(cols)
                    else:
                        fn = list(cols)
                except Exception:
                    fn = list(cols)
                names.extend(fn)
    except Exception:
        pass

    if not names:
        names = fallback_cols
    names = [n.split("__")[-1] for n in names]
    return names

@st.cache_resource
def load_resources():
    model = joblib.load("final_model.pkl")
    pipeline = joblib.load("final_pipeline.pkl")
    df = pd.read_csv("testing-1.csv")
    if "id" not in df.columns:
        raise RuntimeError("testing-1.csv must contain an 'id' column.")

    try:
        dummy = pipeline.transform(df.drop(columns=["id"]).iloc[:1])
        out_dim = to_dense(dummy).shape[1]
        fallback_cols = [f"f_{i}" for i in range(out_dim)]
    except Exception:
        fallback_cols = list(df.drop(columns=["id"]).columns)

    feat_names = safe_feature_names_from_pipeline(pipeline, fallback_cols)
    return model, pipeline, df, feat_names

model, pipeline, test_df, feature_names = load_resources()

def predict_customer(cid: int):
    if cid not in test_df["id"].values:
        return None
    x_df = test_df.loc[test_df["id"] == cid].drop("id", axis=1)
    X = pipeline.transform(x_df)
    prob_d = float(model.predict_proba(X)[0, 1])
    prob_r = 1.0 - prob_d
    risk = "High" if prob_d >= 0.7 else ("Medium" if prob_d >= 0.4 else "Low")
    return prob_d, prob_r, risk, x_df, X

def kernel_shap_single(prepared_X, background_X=None, nsamples=100):
    X_dense = to_dense(prepared_X)
    if background_X is None:
        sample_raw = test_df.drop(columns=["id"]).sample(min(80, len(test_df)), random_state=42)
        background_X = to_dense(pipeline.transform(sample_raw))
    background_X = to_dense(background_X)
    if background_X.shape[0] > 100:
        background_X = background_X[:100, :]

    explainer = shap.KernelExplainer(model.predict_proba, background_X)
    shap_vals = explainer.shap_values(X_dense, nsamples=nsamples)

    if isinstance(shap_vals, list) and len(shap_vals) > 1:
        shap_vals = shap_vals[1]
    shap_vals = to_dense(shap_vals)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)) and len(np.atleast_1d(base_value)) > 1:
        base_value = float(np.atleast_1d(base_value)[1])
    else:
        base_value = float(np.atleast_1d(base_value)[0])

    return shap_vals, base_value, explainer

def align_shap_and_names(shap_vals, names):
    vals = np.nan_to_num(to_dense(shap_vals), nan=0.0, posinf=0.0, neginf=0.0).flatten().astype(float, copy=False)
    n_shap, n_feat = vals.shape[0], len(names)
    if n_shap < n_feat:
        vals = np.pad(vals, (0, n_feat - n_shap))
    elif n_shap > n_feat:
        vals = vals[:n_feat]
    return vals, names

def increasing_decreasing_lists(vals, names, top_k=10, min_abs=1e-8, scale=1.0):
    vals = vals * scale
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    items = [(names[i], float(vals[i])) for i in order if np.isfinite(vals[i])]
    pos = [(f, v) for f, v in items if v > min_abs]
    neg = [(f, v) for f, v in items if v < -min_abs]
    if not pos and not neg:
        items = [(names[i], float(vals[i])) for i in order[:max(5, top_k)]]
        pos = [(f, v) for f, v in items if v >= 0][:5]
        neg = [(f, v) for f, v in items if v < 0][:5]
    return pos, neg, items

def plot_bar_top(items, title):
    if not items:
        st.info("No factors to display.")
        return
    labels = [f for f, _ in items]
    values = [v for _, v in items]
    colors = ["#E53935" if v > 0 else "#43A047" for v in values]
    plt.figure(figsize=(8, 4))
    plt.barh(labels[::-1], values[::-1], color=colors[::-1])
    plt.xlabel("SHAP Impact on Default Probability (approx.)")
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_waterfall_like(vals, names, base_value, title="Waterfall (Force-like)"):
    order = np.argsort(np.abs(vals))[::-1][:10]
    contribs = [(names[i], vals[i]) for i in order]
    labels = [n for n, _ in contribs]
    impacts = [v for _, v in contribs]
    colors = ["#E53935" if v > 0 else "#1E88E5" for v in impacts]

    plt.figure(figsize=(9, 4.5))
    cum = base_value
    left = 0.0
    bars_left, bars_width = [], []
    for v in impacts:
        bars_left.append(left)
        bars_width.append(v)
        left += v

    plt.axvline(base_value, linestyle="--", color="#999999", linewidth=1, label="Base value")
    for y, (l, w, c) in enumerate(zip(bars_left, bars_width, colors)):
        plt.barh([labels[y]], [w], left=l, color=c)

    plt.title(title)
    plt.xlabel("Contribution to Risk Probability (approx.)")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def pretty_factors(factors, limit=5):
    lines = []
    for f, v in factors[:limit]:
        arrow = "↑" if v > 0 else "↓"
        lines.append(f"- {f}: {arrow} ({v:+.6f})")
    return "\n".join(lines) if lines else "No strong factors detected."

# -----------------------------
# Sidebar & Mode
# -----------------------------
st.sidebar.title("🏦 Finlytix Navigation")
mode = st.sidebar.radio("Select Mode", ["📊 Dashboard", "💬 Chatbot"])

# ============================================================
# 📊 DASHBOARD MODE
# ============================================================
if mode == "📊 Dashboard":
    st.title("💰 Finlytix: AI Credit Risk Intelligence")
    st.caption("Transparent, Explainable & Ethical AI for Smart Financial Decisions")

    cid_input = st.text_input("Enter Customer id:")
    if cid_input:
        try:
            cid = int(cid_input)
            result = predict_customer(cid)
            if not result:
                st.error("id not found in dataset.")
            else:
                prob_d, prob_r, risk, x_df, X_single = result
                st.subheader(f"Prediction for Customer id {cid}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Default Probability", f"{prob_d:.2%}")
                c2.metric("Repayment Probability", f"{prob_r:.2%}")
                c3.metric("Risk Level", risk)

                shap_vals_single, base_val, _ = kernel_shap_single(X_single, nsamples=100)
                shap_vals_single, feature_names_aligned = align_shap_and_names(shap_vals_single, feature_names)

                amplify = 1e3
                pos, neg, items = increasing_decreasing_lists(
                    shap_vals_single, feature_names_aligned, top_k=10, min_abs=1e-8, scale=amplify
                )

                st.markdown("### 🔍 Increasing & Decreasing Risk Factors")
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**⬆️ Increasing Risk Factors (higher default chance)**")
                    st.markdown(pretty_factors(pos))
                with cb:
                    st.markdown("**⬇️ Decreasing Risk Factors (safer customer)**")
                    st.markdown(pretty_factors(neg))

                st.markdown("### 📊 Visual SHAP Impact (Top Features)")
                plot_bar_top(items, "Top SHAP Impacts (signed)")

                st.markdown("### ⚡ Waterfall (Force-like) — Local Explanation")
                plot_waterfall_like(np.array(items, dtype=object)[:,1].astype(float),
                                    [it[0] for it in items], base_val, title="Local Waterfall Explanation")

                # ---- Global SHAP (sampled) ----
                st.markdown("---")
                st.markdown("### 🌍 Global SHAP Summary (Sampled)")
                try:
                    raw_sample = test_df.drop(columns=["id"]).sample(min(300, len(test_df)), random_state=42)
                    X_bg = to_dense(pipeline.transform(raw_sample))

                    # TreeExplainer first
                    try:
                        expl = shap.TreeExplainer(model)
                        shap_vals_global = expl.shap_values(X_bg)
                        if isinstance(shap_vals_global, list) and len(shap_vals_global) > 1:
                            shap_vals_global = shap_vals_global[1]
                        shap_vals_global = to_dense(shap_vals_global)
                    except Exception:
                        bg_small = X_bg[:50]
                        expl = shap.KernelExplainer(model.predict_proba, bg_small)
                        shap_vals_global = expl.shap_values(X_bg[:100], nsamples=80)
                        if isinstance(shap_vals_global, list) and len(shap_vals_global) > 1:
                            shap_vals_global = shap_vals_global[1]
                        shap_vals_global = to_dense(shap_vals_global)

                    shap_vals_global = np.nan_to_num(shap_vals_global, nan=0.0)
                    if shap_vals_global.shape[1] != len(feature_names):
                        m = min(shap_vals_global.shape[1], len(feature_names))
                        shap_vals_global = shap_vals_global[:, :m]
                        names_global = feature_names[:m]
                    else:
                        names_global = feature_names

                    mean_abs = np.mean(np.abs(shap_vals_global), axis=0)
                    order = np.argsort(mean_abs)[::-1][:15]
                    labels = [names_global[i] for i in order]
                    values = mean_abs[order] * 1000

                    plt.figure(figsize=(8, 4))
                    plt.barh(labels[::-1], values[::-1], color="#1565C0")
                    plt.xlabel("Mean |SHAP| (scaled)")
                    plt.title("Global Feature Importance (Sampled)")
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.clf()

                    # ---- Fairness ----
                    st.markdown("---")
                    st.markdown("### ⚖️ Fairness Analysis: Influence of MonthlyIncome")
                    income_idx = [i for i, f in enumerate(names_global) if "MonthlyIncome" in f]
                    total = np.sum(mean_abs) if np.sum(mean_abs) > 0 else 1.0
                    income_share = (np.sum(mean_abs[income_idx]) / total * 100.0) if income_idx else 0.0
                    st.write(f"💡 MonthlyIncome contributes **{income_share:.2f}%** of the model's reasoning.")
                    if income_share > 25:
                        st.warning("Model relies heavily on income — review for fairness.")
                    else:
                        st.success("Income influence within fair and ethical range.")

                except Exception as e:
                    st.info("Global SHAP summary unavailable in this environment.")
                    st.text(f"Error: {e}")

                with st.expander("📋 View Customer Data"):
                    st.dataframe(x_df)

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
        m = re.search(r"(\d+)", user_query)
        if not m:
            st.session_state.history.append(("assistant", "Please include a numeric id (e.g., check id 1034)."))
        else:
            cid = int(m.group(1))
            result = predict_customer(cid)
            if not result:
                st.session_state.history.append(("assistant", f"❌ id {cid} not found."))
            else:
                prob_d, prob_r, risk, x_df, X_single = result
                shap_vals_single, base_val, _ = kernel_shap_single(X_single, nsamples=100)
                shap_vals_single, names_aligned = align_shap_and_names(shap_vals_single, feature_names)
                pos, neg, _ = increasing_decreasing_lists(shap_vals_single, names_aligned, top_k=10, min_abs=1e-8, scale=1e3)

                reply = (f"**Customer id {cid}** — Risk: **{risk}**\n\n"
                         f"- Default Probability: **{prob_d:.2%}**\n"
                         f"- Repayment Probability: **{prob_r:.2%}**\n\n"
                         "### 🔍 Increasing Risk Factors\n" + "\n".join([f"- {f}: ↑ ({v:+.6f})" for f, v in pos[:5]]) +
                         "\n\n### 🔍 Decreasing Risk Factors\n" + "\n".join([f"- {f}: ↓ ({v:+.6f})" for f, v in neg[:5]]))
                st.session_state.history.append(("assistant", reply))

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

