#!/usr/bin/env python3
"""
Fix german_credit_processed.csv
Remove 'class' column, keep only 'target'
"""

import pandas as pd

print("Fixing data file...")

df = pd.read_csv('data/german_credit_processed.csv')
print(f"Original: {df.shape}")
print(f"Columns: {list(df.columns)}")

if 'class' in df.columns:
    df = df.drop(columns=['class'])
    print("OK: Dropped 'class' column")

assert 'class' not in df.columns
assert 'target' in df.columns

print(f"\nFixed: {df.shape}")
print(f"Columns: {list(df.columns)}")

df.to_csv('data/german_credit_processed.csv', index=False)
print("\nOK: Saved data/german_credit_processed.csv")
