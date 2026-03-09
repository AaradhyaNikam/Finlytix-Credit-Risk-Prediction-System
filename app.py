import os
import io
import base64
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg') # Required for backend rendering without a display
import matplotlib.pyplot as plt
from scipy import sparse
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# -----------------------------
# Core ML Helpers (From Original)
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
                if name == "remainder": continue
                try:
                    if hasattr(trans, "get_feature_names_out"):
                        fn = list(trans.get_feature_names_out(cols))
                    else:
                        fn = list(cols)
                except:
                    fn = list(cols)
                names.extend(fn)
    except:
        pass
    if not names: names = fallback_cols
    return [n.split("__")[-1] for n in names]

# Load Resources
model = joblib.load("final_model.pkl")
pipeline = joblib.load("final_pipeline.pkl")
test_df = pd.read_csv("testing-1.csv")

try:
    dummy = pipeline.transform(test_df.drop(columns=["id"]).iloc[:1])
    fallback_cols = [f"f_{i}" for i in range(to_dense(dummy).shape[1])]
except:
    fallback_cols = list(test_df.drop(columns=["id"]).columns)
feature_names = safe_feature_names_from_pipeline(pipeline, fallback_cols)



def generate_waterfall_plot(vals, names, base_value):
    order = np.argsort(np.abs(vals))[::-1][:10]
    contribs = [(names[i], vals[i]) for i in order]
    labels = [n for n, _ in contribs]
    impacts = [v for _, v in contribs]
    colors = ["#E53935" if v > 0 else "#1E88E5" for v in impacts]

    # Use Object-Oriented Matplotlib (Thread-safe for Flask)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (l, v) in enumerate(zip(labels, impacts)):
        ax.barh(l, v, color=colors[i])
    
    ax.axvline(base_value, linestyle="--", color="#999999", linewidth=1)
    ax.set_title("Local Waterfall Explanation")
    ax.set_xlabel("Contribution to Risk Probability")
    fig.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)  # Close explicitly to free memory
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    cid = data.get("customer_id")
    
    if not cid or int(cid) not in test_df["id"].values:
        return jsonify({"error": "Invalid or missing Customer ID."}), 400
        
    cid = int(cid)
    x_df = test_df.loc[test_df["id"] == cid].drop("id", axis=1)
    X = pipeline.transform(x_df)
    
    prob_d = float(model.predict_proba(X)[0, 1])
    prob_r = 1.0 - prob_d
    risk = "High" if prob_d >= 0.7 else ("Medium" if prob_d >= 0.4 else "Low")
    
    # Explainability (SHAP)
    X_dense = to_dense(X)
    sample_raw = test_df.drop(columns=["id"]).sample(min(80, len(test_df)), random_state=42)
    background_X = to_dense(pipeline.transform(sample_raw))
    
    explainer = shap.KernelExplainer(model.predict_proba, background_X[:50])
    shap_vals = explainer.shap_values(X_dense, nsamples=50)
    
    if isinstance(shap_vals, list) and len(shap_vals) > 1:
        shap_vals = shap_vals[1]
    
    # 1. Flatten and clean SHAP values
    shap_vals = np.nan_to_num(to_dense(shap_vals), nan=0.0, posinf=0.0, neginf=0.0).flatten().astype(float, copy=False)
    
    # 2. FIX: Align lengths of shap_vals and feature_names to prevent IndexError
    n_shap, n_feat = shap_vals.shape[0], len(feature_names)
    if n_shap < n_feat:
        shap_vals = np.pad(shap_vals, (0, n_feat - n_shap))
    elif n_shap > n_feat:
        shap_vals = shap_vals[:n_feat]
    
    base_val = explainer.expected_value 
    
    base_val = explainer.expected_value
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)): base_val = float(base_val[1])
    
    # Process top features
    order = np.argsort(np.abs(shap_vals))[::-1][:5]
    pos = [{"feature": feature_names[i], "value": float(shap_vals[i])} for i in order if shap_vals[i] > 0]
    neg = [{"feature": feature_names[i], "value": float(shap_vals[i])} for i in order if shap_vals[i] < 0]
    
    # Generate Plot
    plot_b64 = generate_waterfall_plot(shap_vals, feature_names, base_val)

    return jsonify({
        "customer_id": cid,
        "prob_default": f"{prob_d:.2%}",
        "prob_repay": f"{prob_r:.2%}",
        "risk_level": risk,
        "positive_factors": pos,
        "negative_factors": neg,
        "plot_base64": plot_b64,
        "raw_data": x_df.to_dict(orient="records")[0]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)