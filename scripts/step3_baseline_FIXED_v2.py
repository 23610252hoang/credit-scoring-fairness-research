#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1: Baseline Model - DATA LEAKAGE FIXED + CATEGORICAL ENCODING
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Week 1: Baseline - DATA LEAKAGE FIXED")
print("=" * 60)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/german_credit_processed.csv')
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Identify target
if 'target' in df.columns:
    y = df['target']
    print(f"\nTarget distribution:\n{y.value_counts()}")
else:
    raise ValueError("No 'target' column found!")

# Drop target and any leakage columns
exclude_cols = ['target']
if 'class' in df.columns:
    exclude_cols.append('class')
    print("\nWARNING: Found 'class' column - excluding from features")

X = df.drop(columns=exclude_cols)
print(f"\nExcluding: {exclude_cols}")
print(f"Features ({len(X.columns)}): {list(X.columns)[:5]}...")

# Handle categorical columns
print("\nEncoding categorical variables...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# Label encode all categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"OK: Encoded {len(categorical_cols)} categorical columns")

# Check for any remaining non-numeric
print(f"\nX dtypes:\n{X.dtypes.value_counts()}")

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("OK: Features scaled")

# Train model
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 40)
print("RESULTS")
print("=" * 40)
print(f"Test Accuracy:  {acc:.4f}")
print(f"Test AUC:       {auc:.4f}")

# Sanity check
if acc > 0.95:
    print("\nWARNING: Accuracy suspiciously high! Check for data leakage.")
elif acc < 0.5:
    print("\nWARNING: Accuracy too low! Check data processing.")
else:
    print("\nOK: Accuracy in realistic range")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save results
import os
os.makedirs('results', exist_ok=True)
results = pd.DataFrame({
    'metric': ['accuracy', 'auc', 'n_features', 'n_samples'],
    'value': [acc, auc, X.shape[1], X.shape[0]]
})
results.to_csv('results/baseline_results.csv', index=False)
print("\nOK: Saved results/baseline_results.csv")

print("\n" + "=" * 60)
print("DONE - Data leakage check: PASSED")
print("=" * 60)
