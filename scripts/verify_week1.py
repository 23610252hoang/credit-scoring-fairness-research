#!/usr/bin/env python3
"""
Week 1 Verification Script
"""

import sys
import os

def check_python():
    print("\n[1] Python version...")
    v = sys.version_info
    if v.major == 3 and v.minor == 11:
        print(f"OK: Python {v.major}.{v.minor}.{v.micro}")
        return True
    print(f"ERROR: Python {v.major}.{v.minor}.{v.micro} (Need 3.11)")
    return False

def check_deps():
    print("\n[2] Dependencies...")
    deps = ['pandas', 'numpy', 'sklearn', 'certifi']
    ok = True
    for d in deps:
        try:
            __import__(d)
            print(f"OK: {d}")
        except:
            print(f"ERROR: {d}")
            ok = False
    return ok

def check_data():
    print("\n[3] Data file...")
    if not os.path.exists('data/german_credit_processed.csv'):
        print("ERROR: Data file missing")
        return False
    
    import pandas as pd
    df = pd.read_csv('data/german_credit_processed.csv')
    
    print(f"   Shape: {df.shape}")
    
    if 'class' in df.columns:
        print("ERROR: 'class' column exists (LEAKAGE!)")
        return False
    print("OK: NO 'class' column")
    
    if 'target' not in df.columns:
        print("ERROR: 'target' missing")
        return False
    print("OK: 'target' exists")
    
    if len(df.columns) == 21:
        print("OK: 21 columns (20 + target)")
        return True
    print(f"ERROR: {len(df.columns)} columns (expected 21)")
    return False

def check_results():
    print("\n[4] Baseline results...")
    if not os.path.exists('results/baseline_results.csv'):
        print("ERROR: Results missing (run baseline first)")
        return False
    
    import pandas as pd
    df = pd.read_csv('results/baseline_results.csv')
    acc = df['test_accuracy'].values[0]
    
    print(f"   Test Accuracy: {acc:.4f}")
    
    if acc >= 0.99:
        print("ERROR: Accuracy >= 99% (LEAKAGE!)")
        return False
    if 0.70 <= acc <= 0.85:
        print("OK: Realistic accuracy")
        return True
    print("WARNING: Unusual accuracy")
    return True

print("="*60)
print("WEEK 1 VERIFICATION")
print("="*60)

checks = [
    check_python(),
    check_deps(),
    check_data(),
    check_results()
]

print("\n" + "="*60)
if all(checks):
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
print("="*60)
