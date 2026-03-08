#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
池田先生 FIX - Critical Issues Resolution
Fixes: Data leakage, Reproducibility, Git management
"""

import pandas as pd
import os
import sys

print("=" * 70)
print("池田先生 FIX - Critical Issues Resolution")
print("=" * 70)

errors = []

# ============================================================================
# A-1. FIX DATA LEAKAGE - Remove 'class' column from processed data
# ============================================================================
print("\n[A-1] Fixing data leakage in german_credit_processed.csv...")

try:
    df = pd.read_csv('data/german_credit_processed.csv')
    original_cols = list(df.columns)
    print(f"   Original columns ({len(original_cols)}): {original_cols[:5]}...")

    # Check for leakage columns
    leakage_cols = [c for c in ['class', 'Class', 'CLASS'] if c in df.columns]

    if leakage_cols:
        print(f"   WARNING: Found leakage columns: {leakage_cols}")
        df = df.drop(columns=leakage_cols)
        print(f"   OK: Dropped {len(leakage_cols)} leakage column(s)")
    else:
        print("   OK: No 'class' column found")

    # Ensure only one target column exists
    target_cols = [c for c in df.columns if c.lower() in ['target', 'label', 'y']]
    if len(target_cols) > 1:
        print(f"   WARNING: Multiple target columns: {target_cols}")
        # Keep 'target', drop others
        drop_targets = [c for c in target_cols if c != 'target']
        df = df.drop(columns=drop_targets)
        print(f"   OK: Kept only 'target', dropped {drop_targets}")

    # Save cleaned data
    df.to_csv('data/german_credit_processed.csv', index=False)
    print(f"   OK: Saved cleaned data ({df.shape[1]} columns)")

    # Verify
    df_check = pd.read_csv('data/german_credit_processed.csv')
    assert 'class' not in df_check.columns, "class column still exists!"
    assert 'target' in df_check.columns, "target column missing!"
    print("   ✓ Verification passed: No leakage, target exists")

except Exception as e:
    print(f"   ERROR: {e}")
    errors.append(f"Data fix: {e}")

# ============================================================================
# A-2. FIX BASELINE SCRIPT - Add guards and proper exclusion
# ============================================================================
print("\n[A-2] Creating step3_baseline_池田FIXED.py...")

baseline_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1: Baseline - 池田先生 FIX
Data leakage prevented, guards added, realistic accuracy expected
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
print("Week 1: Baseline - 池田先生 FIX")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/german_credit_processed.csv')
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Target
assert 'target' in df.columns, "ERROR: 'target' column not found!"
y = df['target']
print(f"\nTarget distribution:\n{y.value_counts()}")

# CRITICAL: Exclude ALL potential leakage columns
exclude_cols = ['target']  # Always exclude target

# Check for and exclude any 'class' variants
for col in df.columns:
    if col.lower() in ['class', 'label_original']:
        exclude_cols.append(col)
        print(f"WARNING: Excluding potential leakage column '{col}'")

X = df.drop(columns=exclude_cols)

# GUARDS: Ensure no leakage in features
feature_cols = list(X.columns)
print(f"\nFeatures ({len(feature_cols)}): {feature_cols[:5]}...")

# Assertions to prevent data leakage
assert "class" not in feature_cols, "CRITICAL: 'class' in features - data leakage!"
assert "target" not in feature_cols, "CRITICAL: 'target' in features - data leakage!"
assert "Class" not in feature_cols, "CRITICAL: 'Class' in features - data leakage!"

print("✓ Guards passed: No leakage columns in features")

# Encode categoricals
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
print("\n[2] Training Logistic Regression...")
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

# Sanity checks
if acc > 0.95:
    print("\nCRITICAL WARNING: Accuracy suspiciously high! Check for data leakage.")
    print("Expected: 0.70-0.80 for this dataset")
elif acc < 0.5:
    print("\nCRITICAL WARNING: Accuracy too low! Check data processing.")
else:
    print(f"\n✓ Accuracy in realistic range (expected ~0.75)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save results (standardized filename)
import os
os.makedirs('results', exist_ok=True)
results = pd.DataFrame({
    'metric': ['accuracy', 'auc', 'n_features', 'n_samples'],
    'value': [acc, auc, X.shape[1], X.shape[0]]
})
results.to_csv('results/baseline_results.csv', index=False)
print("\n✓ Saved: results/baseline_results.csv")

print("\n" + "=" * 60)
print("DONE - Data leakage check: PASSED")
print("池田先生 FIX applied successfully")
print("=" * 60)
"""

try:
    with open('scripts/step3_baseline_池田FIXED.py', 'w', encoding='utf-8') as f:
        f.write(baseline_script)
    print("   ✓ Created: scripts/step3_baseline_池田FIXED.py")
except Exception as e:
    print(f"   ERROR: {e}")
    errors.append(f"Baseline script: {e}")

# ============================================================================
# B-1. FIX REQUIREMENTS - Add certifi
# ============================================================================
print("\n[B-1] Fixing requirements.txt...")

requirements_content = """# Python 3.11 recommended (tested on 3.11.9)
# certifi required for SSL certificate verification
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
certifi>=2023.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

try:
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    print("   ✓ Created: requirements.txt (with certifi)")

    # Remove old requirement.txt if exists
    if os.path.exists('requirement.txt'):
        os.remove('requirement.txt')
        print("   ✓ Removed old requirement.txt")

except Exception as e:
    print(f"   ERROR: {e}")
    errors.append(f"Requirements: {e}")

# ============================================================================
# B-2. FIX README - Python version and instructions
# ============================================================================
print("\n[B-2] Creating README_池田FIXED.md...")

readme_content = """# Credit Fairness Analysis - 池田Lab Workshop 2026

## Environment Setup (池田先生 FIX applied)

### Requirements
- **Python 3.11.x** (tested on 3.11.9, conda recommended)
- See `requirements.txt` for dependencies

### Setup Steps

```bash
# 1. Create conda environment (recommended)
conda create -n credit-fairness python=3.11.9
conda activate credit-fairness

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import pandas, numpy, sklearn, certifi; print('OK')"
```

## Week 1: Baseline (池田先生 FIX)

```bash
# Download data
python scripts/step1_download_data_FIXED.py

# Run baseline (NO data leakage, realistic accuracy ~0.75)
python scripts/step3_baseline_池田FIXED.py

# Verify results
python scripts/verify_week1_FINAL.py
```

**Expected Results:**
- Test Accuracy: ~0.72-0.76 (realistic)
- Test AUC: ~0.75-0.79
- NO 'class' column in features (leakage prevented)

## Week 2: Fairness Analysis

```bash
python scripts/step4_fairness_analysis.py
```

## Week 3: Bias Mitigation

```bash
python scripts/step5_bias_mitigation.py
```

## Repository Structure

```
├── data/                    # Data files (not tracked by git)
├── scripts/                 # Python scripts
├── results/                 # Output CSVs (not tracked by git)
├── figs/                    # Generated figures (not tracked by git)
├── docs/                    # Documentation
├── requirements.txt         # Dependencies (池田先生 FIX)
└── README.md               # This file
```

## Git Management (池田先生 FIX)

- `data/`, `results/`, `figs/` are excluded from git tracking
- Only source code and documentation are version controlled
"""

try:
    with open('README_池田FIXED.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("   ✓ Created: README_池田FIXED.md")
except Exception as e:
    print(f"   ERROR: {e}")

# ============================================================================
# C-1. FIX GITIGNORE - Stop tracking generated files
# ============================================================================
print("\n[C-1] Fixing .gitignore...")

gitignore_content = """# Data files
data/*.csv
data/*.json
data/*.xlsx

# Results
results/*.csv
results/*.json
results/*.txt

# Figures (generated)
figs/*.png
figs/*.jpg
figs/*.pdf
figs/*.svg

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Exceptions: Keep docs and sample data if needed
!docs/*.md
!docs/*.pdf
!data/.gitkeep
!results/.gitkeep
!figs/.gitkeep
"""

try:
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("   ✓ Created: .gitignore (properly excludes data/results/figs)")
    print("   ⚠ Run 'git rm -r --cached data results figs' to stop tracking")
except Exception as e:
    print(f"   ERROR: {e}")
    errors.append(f"Gitignore: {e}")

# ============================================================================
# C-2. CLEANUP OLD FILES
# ============================================================================
print("\n[C-2] Cleaning up old/duplicate files...")

files_to_remove = [
    'scripts/step3_baseline_FIXED.py',  # Old version with potential issues
    'results/baseline_results_corrected.csv',  # Duplicate naming
]

for f in files_to_remove:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"   ✓ Removed: {f}")
        except Exception as e:
            print(f"   ⚠ Could not remove {f}: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("池田先生 FIX - SUMMARY")
print("=" * 70)

if not errors:
    print("\n✅ ALL FIXES APPLIED SUCCESSFULLY")
    print("\nNext steps:")
    print("1. Run: python scripts/step3_baseline_池田FIXED.py")
    print("2. Verify: python scripts/verify_week1_FINAL.py")
    print("3. Git cleanup: git rm -r --cached data results figs")
    print("4. Commit: git add -A && git commit -m '池田先生 FIX: Data leakage, reproducibility'")
    print("5. Push: git push origin main")
    print("\nExpected result: Accuracy ~0.75 (NOT 1.0)")
else:
    print("\n⚠ SOME FIXES FAILED:")
    for e in errors:
        print(f"   - {e}")

print("=" * 70)
