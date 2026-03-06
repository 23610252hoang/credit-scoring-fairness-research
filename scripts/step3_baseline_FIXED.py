#!/usr/bin/env python3
"""
Week 1: Baseline Model - DATA LEAKAGE FIXED
Fixed: Exclude BOTH 'class' and 'target' from features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import os

print("="*60)
print("Week 1: Baseline - DATA LEAKAGE FIXED")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/german_credit_processed.csv')
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# CRITICAL FIX: Exclude BOTH 'class' and 'target'
exclude_cols = ['class', 'target']  # <- FIXED!

print(f"\nExcluding: {exclude_cols}")
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")

# SAFETY CHECKS
assert 'class' not in feature_cols, "ERROR: 'class' leaked!"
assert 'target' not in feature_cols, "ERROR: 'target' leaked!"
print("OK: Data leakage check PASSED")

# Prepare X, y
X = df[feature_cols].values
y = df['target'].values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

print("\n" + "="*60)
print("RESULTS (LEAKAGE FIXED)")
print("="*60)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUC:      {train_auc:.4f}")
print(f"Test AUC:       {test_auc:.4f}")

# Sanity check
if train_acc >= 0.99 or test_acc >= 0.99:
    print("\nWARNING: Accuracy >= 99% - Possible leakage!")
else:
    print("\nOK: Accuracy in realistic range")

# Save
os.makedirs('results', exist_ok=True)
results = pd.DataFrame([{
    'model': 'Logistic Regression',
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'train_auc': train_auc,
    'test_auc': test_auc
}])
results.to_csv('results/baseline_results.csv', index=False)

print("\nOK: Saved results/baseline_results.csv")
print("="*60)
