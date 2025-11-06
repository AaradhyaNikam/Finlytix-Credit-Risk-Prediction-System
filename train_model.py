# =====================================
# Finlytix - Model Training Script
# =====================================

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# File paths
# -------------------------------
TRAIN_FILE = "training.csv"
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# -------------------------------
# Load training dataset
# -------------------------------
data = pd.read_csv(TRAIN_FILE)

# Create income categories (bins)
data['income_cat'] = pd.cut(
    data['MonthlyIncome'],
    bins=[-1, 3000, 6000, 9000, 12000, np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Drop NaN in income_cat and convert to int
data = data.dropna(subset=['income_cat']).copy()
data['income_cat'] = data['income_cat'].astype(int)

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = data.loc[test_index].drop("income_cat", axis=1)

# Split into features and labels
train_features = strat_train_set.drop("SeriousDlqin2yrs", axis=1)
train_labels = strat_train_set["SeriousDlqin2yrs"].astype(int)

# -------------------------------
# Preprocessing pipeline
# -------------------------------
num_attributes = train_features.select_dtypes(include=[np.number]).columns.to_list()
cat_attributes = train_features.select_dtypes(exclude=[np.number]).columns.to_list()

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

train_prepared = full_pipeline.fit_transform(train_features)

# -------------------------------
# Train XGBoost Model
# -------------------------------
XGB = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

XGB.fit(train_prepared, train_labels)

# Cross-validation
auc_scores = cross_val_score(XGB, train_prepared, train_labels, scoring="roc_auc", cv=5)
print(f"Cross-validation ROC-AUC Mean: {auc_scores.mean():.4f}")

# -------------------------------
# Save model and pipeline
# -------------------------------
joblib.dump(XGB, MODEL_FILE)
joblib.dump(full_pipeline, PIPELINE_FILE)
print("✅ Model and preprocessing pipeline saved successfully!")

# Evaluate on test split
test_features = strat_test_set.drop("SeriousDlqin2yrs", axis=1)
test_labels = strat_test_set["SeriousDlqin2yrs"].astype(int)
test_prepared = full_pipeline.transform(test_features)

y_pred = XGB.predict(test_prepared)
y_prob = XGB.predict_proba(test_prepared)[:, 1]
auc = roc_auc_score(test_labels, y_prob)

print(f"🎯 Test ROC-AUC: {auc:.4f}")
print("\nClassification Report:\n", classification_report(test_labels, y_pred))
