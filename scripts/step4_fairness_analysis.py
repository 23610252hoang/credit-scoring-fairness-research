#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 2: Fairness Analysis
Analyze bias in credit scoring by sex and age groups
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Week 2: Fairness Analysis")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/german_credit_processed.csv')
print(f"Data shape: {df.shape}")

# Identify protected attributes
protected_attrs = []
if 'sex' in df.columns:
    protected_attrs.append('sex')
elif 'sex_binary' in df.columns:
    protected_attrs.append('sex_binary')

if 'age' in df.columns:
    protected_attrs.append('age')
elif 'age_binary' in df.columns:
    protected_attrs.append('age_binary')

print(f"Protected attributes: {protected_attrs}")

# Prepare data (same as baseline)
y = df['target']
exclude_cols = ['target']
if 'class' in df.columns:
    exclude_cols.append('class')

X = df.drop(columns=exclude_cols)

# Encode categorical
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Split and train (same random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n[2] Overall Performance")
print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")

# Get test indices to match with original df
test_indices = X_test.index

# Fairness by Sex
print("\n" + "=" * 60)
print("[3] FAIRNESS BY SEX")
print("=" * 60)

sex_col = None
if 'sex_binary' in df.columns:
    sex_col = 'sex_binary'
elif 'sex' in df.columns:
    sex_col = 'sex'

if sex_col:
    sex_test = df.loc[test_indices, sex_col]

    for sex_val in sorted(sex_test.unique()):
        mask = sex_test == sex_val
        group_name = "Male" if sex_val == 1 else "Female" if sex_val == 0 else f"Sex={sex_val}"

        y_true_group = y_test[mask]
        y_pred_group = y_pred[mask]
        y_prob_group = y_prob[mask]

        acc = np.mean(y_pred_group == y_true_group)
        approval_rate = np.mean(y_pred_group)

        print(f"\n{group_name} (n={sum(mask)}):")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Approval Rate: {approval_rate:.4f}")
        print(f"  Actual Default Rate: {1-np.mean(y_true_group):.4f}")

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

        print(f"  TPR (Sensitivity): {tpr:.4f}")
        print(f"  FPR (False Alarm): {fpr:.4f}")

# Fairness by Age
print("\n" + "=" * 60)
print("[4] FAIRNESS BY AGE GROUP")
print("=" * 60)

age_col = None
if 'age_binary' in df.columns:
    age_col = 'age_binary'
elif 'age_group' in df.columns:
    age_col = 'age_group'
elif 'age' in df.columns:
    age_col = 'age'

if age_col:
    age_test = df.loc[test_indices, age_col]

    # If continuous age, bin it
    if age_col == 'age':
        age_test = (age_test >= 25).astype(int)
        age_labels = {0: "Young (<25)", 1: "Old (>=25)"}
    else:
        age_labels = {0: "Young", 1: "Old"}

    for age_val in sorted(age_test.unique()):
        mask = age_test == age_val
        group_name = age_labels.get(age_val, f"Age={age_val}")

        y_true_group = y_test[mask]
        y_pred_group = y_pred[mask]

        acc = np.mean(y_pred_group == y_true_group)
        approval_rate = np.mean(y_pred_group)

        print(f"\n{group_name} (n={sum(mask)}):")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Approval Rate: {approval_rate:.4f}")

        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"  TPR: {tpr:.4f}")
        print(f"  FPR: {fpr:.4f}")

# Calculate Fairness Metrics
print("\n" + "=" * 60)
print("[5] FAIRNESS METRICS SUMMARY")
print("=" * 60)

if sex_col and age_col:
    sex_test = df.loc[test_indices, sex_col]
    age_test = df.loc[test_indices, age_col]
    if age_col == 'age':
        age_test = (age_test >= 25).astype(int)

    # Demographic Parity: Approval rates should be similar across groups
    print("\nDemographic Parity (Approval Rates):")
    male_approve = np.mean(y_pred[sex_test == 1])
    female_approve = np.mean(y_pred[sex_test == 0])
    print(f"  Male: {male_approve:.4f}, Female: {female_approve:.4f}")
    print(f"  Disparate Impact: {female_approve/male_approve:.4f} (1.0 = fair, <0.8 = bias)")

    old_approve = np.mean(y_pred[age_test == 1])
    young_approve = np.mean(y_pred[age_test == 0])
    print(f"  Old: {old_approve:.4f}, Young: {young_approve:.4f}")
    print(f"  Disparate Impact: {young_approve/old_approve:.4f}")

    # Equalized Odds: TPR and FPR should be similar
    print("\nEqualized Odds (TPR/FPR by Sex):")

    def calc_tpr_fpr(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return tpr, fpr

    male_tpr, male_fpr = calc_tpr_fpr(y_test[sex_test == 1], y_pred[sex_test == 1])
    female_tpr, female_fpr = calc_tpr_fpr(y_test[sex_test == 0], y_pred[sex_test == 0])

    print(f"  Male   - TPR: {male_tpr:.4f}, FPR: {male_fpr:.4f}")
    print(f"  Female - TPR: {female_tpr:.4f}, FPR: {female_fpr:.4f}")
    print(f"  TPR Ratio: {female_tpr/male_tpr:.4f} (1.0 = fair)")

# Save results
import os
os.makedirs('results', exist_ok=True)

results_summary = {
    'metric': [],
    'group': [],
    'value': []
}

# Add sex metrics
if sex_col:
    for sex_val in [0, 1]:
        mask = sex_test == sex_val
        group_name = "Male" if sex_val == 1 else "Female"
        results_summary['metric'].append('approval_rate')
        results_summary['group'].append(group_name)
        results_summary['value'].append(np.mean(y_pred[mask]))

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/fairness_metrics.csv', index=False)
print("\n[6] Saved: results/fairness_metrics.csv")

print("\n" + "=" * 60)
print("Week 2 Analysis Complete")
print("=" * 60)
