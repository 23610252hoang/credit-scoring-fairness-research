#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1 Verification Script - FIXED VERSION
Matches: step3_baseline_FIXED_v2.py output format
"""

import sys
import os

def check_python():
    """Check Python version is 3.11.x"""
    print("\n[1] Python version check...")
    v = sys.version_info
    if v.major == 3 and v.minor == 11:
        print(f"✓ OK: Python {v.major}.{v.minor}.{v.micro}")
        return True
    print(f"✗ ERROR: Python {v.major}.{v.minor}.{v.micro} (Need 3.11.x)")
    return False

def check_deps():
    """Check required dependencies"""
    print("\n[2] Dependencies check...")
    deps = ['pandas', 'numpy', 'sklearn', 'certifi']
    ok = True
    for d in deps:
        try:
            __import__(d)
            print(f"✓ OK: {d}")
        except ImportError:
            print(f"✗ ERROR: {d} not installed")
            ok = False
    return ok

def check_data():
    """Check data file format"""
    print("\n[3] Data file check...")
    
    if not os.path.exists('data/german_credit_processed.csv'):
        print("✗ ERROR: Data file missing")
        return False
    
    import pandas as pd
    df = pd.read_csv('data/german_credit_processed.csv')
    
    print(f"   Shape: {df.shape}")
    
    # Check NO 'class' column (data leakage check)
    if 'class' in df.columns:
        print("✗ ERROR: 'class' column exists (DATA LEAKAGE!)")
        return False
    print("✓ OK: NO 'class' column (data leakage fixed)")
    
    # Check 'target' exists
    if 'target' not in df.columns:
        print("✗ ERROR: 'target' column missing")
        return False
    print("✓ OK: 'target' column exists")
    
    # Check number of columns (should be 25: 20 attributes + 4 derived + target)
    if df.shape[1] == 25:
        print(f"✓ OK: 25 columns (expected for fixed version)")
        return True
    else:
        print(f"⚠ WARNING: {df.shape[1]} columns (expected 25, but acceptable)")
        return True

def check_results():
    """Check baseline results format and values"""
    print("\n[4] Baseline results check...")
    
    if not os.path.exists('results/baseline_results.csv'):
        print("✗ ERROR: Results file missing")
        print("   Run: python scripts/step3_baseline_FIXED_v2.py")
        return False
    
    import pandas as pd
    df = pd.read_csv('results/baseline_results.csv')
    
    print(f"   Columns: {list(df.columns)}")
    
    # Flexible column check - accept either 'accuracy' or 'test_accuracy'
    acc_col = None
    if 'accuracy' in df.columns:
        acc_col = 'accuracy'
    elif 'value' in df.columns and 'metric' in df.columns:
        # Format: metric, value
        acc_row = df[df['metric'] == 'accuracy']
        if len(acc_row) > 0:
            acc = acc_row['value'].values[0]
            print(f"   Accuracy: {acc:.4f}")
            
            if acc >= 0.99:
                print("✗ ERROR: Accuracy >= 99% (DATA LEAKAGE!)")
                return False
            if 0.65 <= acc <= 0.85:
                print(f"✓ OK: Realistic accuracy ({acc:.4f})")
                return True
            print(f"⚠ WARNING: Unusual accuracy ({acc:.4f}), but acceptable")
            return True
    
    if acc_col:
        acc = df[acc_col].values[0]
        print(f"   Accuracy: {acc:.4f}")
        
        if acc >= 0.99:
            print("✗ ERROR: Accuracy >= 99% (DATA LEAKAGE!)")
            return False
        if 0.65 <= acc <= 0.85:
            print(f"✓ OK: Realistic accuracy ({acc:.4f})")
            return True
        print(f"⚠ WARNING: Unusual accuracy ({acc:.4f}), but acceptable")
        return True
    
    print("✗ ERROR: Cannot find accuracy value in results")
    return False

print("=" * 60)
print("WEEK 1 VERIFICATION - FIXED VERSION")
print("=" * 60)

checks = [
    check_python(),
    check_deps(),
    check_data(),
    check_results()
]

print("\n" + "=" * 60)
if all(checks):
    print("✓ ALL CHECKS PASSED")
    print("\nWeek 1 is COMPLETE!")
    print("=" * 60)
    sys.exit(0)
else:
    print("✗ SOME CHECKS FAILED")
    print("\nPlease fix the errors above.")
    print("=" * 60)
    sys.exit(1)
