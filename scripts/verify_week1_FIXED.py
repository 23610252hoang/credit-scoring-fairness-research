#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1 Verification Script - FIXED
"""

import sys
import subprocess
import pandas as pd
import os

print("=" * 60)
print("WEEK 1 VERIFICATION")
print("=" * 60)

# [1] Python version
print("\n[1] Python version...")
version = sys.version_info
print(f"   Python {version.major}.{version.minor}.{version.micro}")

# FIXED: Accept Python 3.10+
if version.major == 3 and version.minor >= 10:
    print(f"OK: Python 3.{version.minor} (>= 3.10 accepted)")
else:
    print(f"WARNING: Python {version.major}.{version.minor} (Recommend 3.10+)")

# [2] Dependencies
print("\n[2] Dependencies...")
required = ['pandas', 'numpy', 'sklearn', 'certifi']
for pkg in required:
    try:
        __import__(pkg.replace('sklearn', 'sklearn'))
        print(f"OK: {pkg}")
    except ImportError:
        print(f"ERROR: {pkg} not installed")

# [3] Data file
print("\n[3] Data file...")
try:
    df = pd.read_csv('data/german_credit_processed.csv')
    print(f"   Shape: {df.shape}")

    # Check columns
    cols = list(df.columns)

    if 'class' in cols:
        print("ERROR: 'class' column exists (data leakage!)")
    else:
        print("OK: NO 'class' column")

    if 'target' in cols:
        print("OK: 'target' exists")
    else:
        print("ERROR: 'target' not found")

    # FIXED: Accept 21-25 columns (original + derived features)
    n_cols = len(cols)
    if 21 <= n_cols <= 25:
        print(f"OK: {n_cols} columns (expected 21-25)")
    else:
        print(f"WARNING: {n_cols} columns (unexpected)")

except Exception as e:
    print(f"ERROR: {e}")

# [4] Baseline results
print("\n[4] Baseline results...")
try:
    if os.path.exists('results/baseline_results.csv'):
        df_res = pd.read_csv('results/baseline_results.csv')
        print(f"   Columns: {list(df_res.columns)}")

        # FIXED: Check both possible column names
        if 'test_accuracy' in df_res.columns:
            acc = df_res['test_accuracy'].values[0]
        elif 'accuracy' in df_res.columns:
            acc = df_res['accuracy'].values[0]
        else:
            print("ERROR: No accuracy column found")
            acc = None

        if acc is not None:
            print(f"   Test Accuracy: {acc:.4f}")

            if 0.65 <= acc <= 0.85:
                print("OK: Realistic accuracy")
            elif acc > 0.95:
                print("ERROR: Accuracy too high (data leakage?)")
            else:
                print("WARNING: Accuracy unusual")
    else:
        print("WARNING: No results file found")
        print("   Run: python scripts/step3_baseline_FIXED_v2.py")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
