#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1 Verification Script - FINAL VERSION
"""

import sys
import pandas as pd
import os

print("=" * 60)
print("WEEK 1 VERIFICATION - FINAL")
print("=" * 60)

all_pass = True

# [1] Python version
print("\n[1] Python version...")
version = sys.version_info
print(f"   Python {version.major}.{version.minor}.{version.micro}")

if version.major == 3 and version.minor >= 10:
    print(f"OK: Python 3.{version.minor} (>= 3.10 accepted)")
else:
    print(f"WARNING: Python {version.major}.{version.minor} (Recommend 3.10+)")
    all_pass = False

# [2] Dependencies
print("\n[2] Dependencies...")
required = {
    'pandas': 'pd',
    'numpy': 'np', 
    'sklearn': 'sklearn',
    'certifi': 'certifi'
}

for pkg, import_name in required.items():
    try:
        __import__(import_name)
        print(f"OK: {pkg}")
    except ImportError:
        print(f"ERROR: {pkg} not installed")
        all_pass = False

# [3] Data file
print("\n[3] Data file...")
try:
    df = pd.read_csv('data/german_credit_processed.csv')
    print(f"   Shape: {df.shape}")
    cols = list(df.columns)

    if 'class' in cols:
        print("ERROR: 'class' column exists (data leakage!)")
        all_pass = False
    else:
        print("OK: NO 'class' column")

    if 'target' in cols:
        print("OK: 'target' exists")
    else:
        print("ERROR: 'target' not found")
        all_pass = False

    # Accept 21-25 columns
    n_cols = len(cols)
    if 21 <= n_cols <= 25:
        print(f"OK: {n_cols} columns (expected 21-25)")
    else:
        print(f"WARNING: {n_cols} columns (unexpected)")

except Exception as e:
    print(f"ERROR: {e}")
    all_pass = False

# [4] Baseline results - FIXED: Read metric,value format
print("\n[4] Baseline results...")
try:
    if os.path.exists('results/baseline_results.csv'):
        df_res = pd.read_csv('results/baseline_results.csv')
        print(f"   Format: {list(df_res.columns)}")

        # Handle metric,value format
        if 'metric' in df_res.columns and 'value' in df_res.columns:
            # Format: metric,value
            metrics_dict = dict(zip(df_res['metric'], df_res['value']))
            acc = metrics_dict.get('accuracy')
            auc = metrics_dict.get('auc')
            print(f"   Accuracy: {acc}")
            print(f"   AUC: {auc}")
        elif 'accuracy' in df_res.columns:
            # Format: accuracy,auc columns
            acc = df_res['accuracy'].values[0]
            auc = df_res.get('auc', [None])[0]
            print(f"   Accuracy: {acc}")
        else:
            print("ERROR: Unknown format")
            acc = None

        if acc is not None:
            print(f"   Test Accuracy: {acc:.4f}")

            if 0.65 <= acc <= 0.85:
                print("OK: Realistic accuracy (0.65-0.85)")
            elif acc > 0.95:
                print("ERROR: Accuracy too high - data leakage suspected!")
                all_pass = False
            elif acc < 0.5:
                print("ERROR: Accuracy too low")
                all_pass = False
            else:
                print(f"WARNING: Accuracy {acc:.4f} outside normal range")
    else:
        print("WARNING: results/baseline_results.csv not found")
        print("   Run: python scripts/step3_baseline_FIXED_v2.py")
        all_pass = False
except Exception as e:
    print(f"ERROR: {e}")
    all_pass = False

# Summary
print("\n" + "=" * 60)
if all_pass:
    print("ALL CHECKS PASSED - READY FOR WEEK 2")
else:
    print("SOME CHECKS FAILED - SEE ERRORS ABOVE")
print("=" * 60)
