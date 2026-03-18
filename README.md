# Credit Fairness Analysis

**Student:** Hoang Nguyen  
**University:** Yamato University, Faculty of Informatics  
**Advisor:** Prof. Ikeda

---

## 🚨 CRITICAL FIX (2026-03-06 - Workshop1 Verified)

**Data leakage FIXED:**
- **Previous:** 'class' column leaked → Accuracy = 100%
- **Fixed:** 'class' excluded from features
- **Now:** Realistic accuracy (~72.5%)
- **Verified on:** workshop1 (Python 3.11 conda environment)

---

## Prerequisites

### Python Version: 3.11.x (REQUIRED)

```bash
python --version  # Must be 3.11.x
```

**Why 3.11.x?** pandas installation fails on Python 3.13 venv

### Setup

```bash
# Conda (recommended)
conda create -n credit python=3.11
conda activate credit

# OR venv
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Week 1: Baseline Models

```bash
# Step 1: Preprocess data
python scripts/fix_data.py

# Step 2: Run baseline model
python scripts/step3_baseline_FIXED_v2.py

# Step 3: Verify results
python scripts/verify_week1.py
```

### Expected Output

```
Test Accuracy:  0.7250
Test AUC:       0.7564
```

**Verify should show:**
```
✓ ALL CHECKS PASSED
Week 1 is COMPLETE!
```

---

## Project Structure

```
nguyen-credit-fairness-2026/
├── README.md
├── requirements.txt           <- Dependencies (Python 3.11.x)
│
├── data/
│   └── german_credit_processed.csv  <- 25 columns (NO 'class')
│
├── scripts/
│   ├── fix_data.py                  <- Data preprocessing
│   ├── step3_baseline_FIXED_v2.py   <- Week 1 OFFICIAL version
│   └── verify_week1.py              <- Verification (FIXED)
│
├── old/                       <- Archived old versions
│   ├── step3_baseline.py              (original - with leakage)
│   ├── step3_baseline_FIXED.py        (intermediate version)
│   └── step3_baseline_ikeda.py        (professor's reference)
│
└── results/
    └── baseline_results.csv   <- Format: metric, value
```

---

## Verification

### Run Verification Script

```bash
python scripts/verify_week1.py
```

### Checks Performed

```
[1] Python version
    ✓ 3.11.x (strict requirement)

[2] Dependencies
    ✓ pandas, numpy, scikit-learn, certifi

[3] Data file
    ✓ NO 'class' column (data leakage prevention)
    ✓ 'target' column exists
    ✓ 25 columns (flexible check)

[4] Results
    ✓ Accuracy NOT 1.0 (leakage indicator)
    ✓ Realistic range: 0.65-0.85
```

---

## Key Results

### Current (FIXED - 2026-03-06)

**Model:** Logistic Regression (Baseline)

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.7250 |
| Test AUC | 0.7564 |
| Data Leakage | **FIXED** ✓ |

### Previous (BROKEN - Had Data Leakage)

| Metric | Value |
|--------|-------|
| Test Accuracy | 1.0000 ⚠️ |
| Test AUC | 1.0000 ⚠️ |
| Cause | 'class' column in features |

---

## Data Leakage Fix Details

### Problem

Original data had both `class` and `target` columns with identical values:
```python
# BAD: Model could just copy the answer
X = df.drop(columns=['target'])  # Still has 'class'!
→ Accuracy = 100%
```

### Solution

```python
# GOOD: Exclude both 'class' and 'target'
exclude_cols = ['target']
if 'class' in df.columns:
    exclude_cols.append('class')
    
X = df.drop(columns=exclude_cols)
→ Accuracy = 72.5% (realistic)
```

### Verification

```bash
# Check NO 'class' column
python -c "import pandas as pd; df = pd.read_csv('data/german_credit_processed.csv'); print('class' in df.columns)"
# Output: False ✓
```

---

## Troubleshooting

### Python Version Error

```
ERROR: Python 3.13.0 (Need 3.11.x)
```

**Solution:** Use Python 3.11.x
```bash
conda create -n credit python=3.11
conda activate credit
pip install -r requirements.txt
```

### Column Count Warning

```
WARNING: X columns (expected 21)
```

**Solution:** This is normal. Current version has 25 columns:
- 20 original attributes
- 1 target
- 4 derived features (age_binary, age_group, sex_binary, sex_group)

Verification script now uses **flexible column check** (not fixed to 21).

### Dependencies Installation Fails

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Repository

**GitHub:** https://github.com/ikedalab-org/nguyen-credit-fairness-2026

---

## Contact

**Hoang Nguyen**  
Yamato University, Faculty of Informatics  
Advisor: Prof. Ikeda

---

## Changelog

### 2026-03-06 (Workshop1 Verification)
- ✅ Fixed: requirements.txt filename standardized
- ✅ Fixed: verify_week1.py updated for flexible checks
- ✅ Verified: Data leakage completely resolved
- ✅ Verified: Reproducibility on workshop1 (Python 3.11)

### 2026-03-05 (Initial Fix)
- ✅ Fixed: Data leakage by excluding 'class' column
- ✅ Added: Verification script
- ✅ Updated: All scripts to use fixed data
