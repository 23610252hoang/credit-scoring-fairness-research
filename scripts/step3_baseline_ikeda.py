#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

print('Week 1: Baseline - 池田先生 FIX')
print('=' * 50)

df = pd.read_csv('data/german_credit_processed.csv')
y = df['target']
X = df.drop(columns=['target'])

assert 'class' not in X.columns
print('OK: No data leakage')

for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f'Accuracy: {acc:.4f}')
print(f'AUC: {auc:.4f}')

if 0.65 <= acc <= 0.85:
    print('OK: Realistic accuracy')
elif acc > 0.95:
    print('ERROR: Accuracy too high!')

os.makedirs('results', exist_ok=True)
pd.DataFrame({'metric': ['accuracy', 'auc'], 'value': [acc, auc]}).to_csv('results/baseline_results.csv', index=False)
print('Saved: results/baseline_results.csv')
