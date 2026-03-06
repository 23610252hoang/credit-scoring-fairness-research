# Credit Fairness Analysis

**Student:** Hoang Nguyen  
**University:** Yamato University, Faculty of Informatics

---

## CRITICAL FIX (2026-03-05)

**Data leakage FIXED:**
- Previous: 'class' column leaked -> 100% accuracy
- Fixed: Both 'class' and 'target' excluded from features
- Now: Realistic accuracy (~76%)

---

## Prerequisites

**Python Version:** 3.11.x (REQUIRED)

```bash
python --version  # Must be 3.11.x
```

**Setup:**
```bash
# Conda (recommended)
conda create -n credit python=3.11
conda activate credit

# OR venv
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

**Install:**
```bash
pip install -r requirements.txt
```

---

## Quick Start

### Step 1: Fix Data
```bash
python scripts/fix_data.py
```

### Step 2: Run Baseline
```bash
python scripts/step3_baseline_FIXED.py
```

**Expected output:**
- Test Accuracy: ~0.76 (NOT 1.0!)
- Results: `results/baseline_results.csv`

### Step 3: Verify
```bash
python scripts/verify_week1.py
```

---

## Project Structure

```
scripts/
  step1_download_data_FIXED.py
  step3_baseline_FIXED.py    <- LEAKAGE FIXED
  fix_data.py                <- Remove 'class'
  verify_week1.py            <- Verification
data/
  german_credit_processed.csv  <- 21 cols (NO 'class')
results/
  baseline_results.csv
requirements.txt             <- With certifi
README.md
```

---

## Verification

Run verification:
```bash
python scripts/verify_week1.py
```

Should show:
- OK: Python 3.11
- OK: All dependencies
- OK: NO 'class' column
- OK: Accuracy realistic (NOT 1.0)

---

## Key Results

**Baseline (Fixed):**
- Test Accuracy: ~0.763
- Test AUC: ~0.783

**Previous (Broken):**
- Test Accuracy: 1.000 <- DATA LEAKAGE!

---

## Contact

Hoang Nguyen  
Yamato University, Faculty of Informatics
