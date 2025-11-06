# =====================================
# Finlytix - Final Production Model Training Script
# =====================================

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# File Paths
# -------------------------------
TRAIN_FILE = "training.csv"
MODEL_FILE = "final_model.pkl"
PIPELINE_FILE = "final_pipeline.pkl"

# -------------------------------
# Load Full Training Dataset
# -------------------------------
data = pd.read_csv(TRAIN_FILE)
print(f"✅ Loaded training data with shape: {data.shape}")

# Split into features and labels
features = data.drop("SeriousDlqin2yrs", axis=1)
labels = data["SeriousDlqin2yrs"].astype(int)

# -------------------------------
# Preprocessing Pipelines
# -------------------------------
num_attributes = features.select_dtypes(include=[np.number]).columns.to_list()
cat_attributes = features.select_dtypes(exclude=[np.number]).columns.to_list()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
], sparse_threshold=0.3)

# Transform the full dataset
prepared_features = full_pipeline.fit_transform(features)
print(f"✅ Data preprocessed successfully. Shape: {prepared_features.shape}")

# -------------------------------
# Train Final XGBoost Model
# -------------------------------
XGB_final = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

XGB_final.fit(prepared_features, labels)
print("✅ Final model trained successfully on full dataset!")

# -------------------------------
# Save Final Model & Pipeline
# -------------------------------
joblib.dump(XGB_final, MODEL_FILE)
joblib.dump(full_pipeline, PIPELINE_FILE)
print("💾 Final model and preprocessing pipeline saved as 'final_model.pkl' and 'final_pipeline.pkl'")
