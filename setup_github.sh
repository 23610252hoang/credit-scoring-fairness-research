#!/bin/bash
# Setup GitHub Repo - 池田先生 FIX
# Chạy từ thư mục gốc project

echo "=========================================="
echo "Setup GitHub Repo - 池田先生 FIX"
echo "=========================================="

# 1. Tạo cấu trúc thư mục
echo "[1] Tạo thư mục..."
mkdir -p scripts
mkdir -p data
mkdir -p results
mkdir -p figs
mkdir -p docs

# 2. Tạo .gitkeep để giữ thư mục trống
touch data/.gitkeep
touch results/.gitkeep
touch figs/.gitkeep

# 3. Copy các script vào thư mục scripts/
echo "[2] Copy scripts..."
cp step3_baseline_ikeda.py scripts/
cp step4_fairness_analysis.py scripts/ 2>/dev/null || echo "  (Chưa có step4)"
cp step5_bias_mitigation.py scripts/ 2>/dev/null || echo "  (Chưa có step5)"
cp verify_week1_FINAL.py scripts/ 2>/dev/null || echo "  (Chưa có verify)"

# 4. Tạo .gitignore
echo "[3] Tạo .gitignore..."
cat > .gitignore << 'EOF'
# Data files - DO NOT TRACK
data/*.csv
data/*.json
data/*.xlsx

# Results - DO NOT TRACK
results/*.csv
results/*.json
results/*.txt

# Figures - DO NOT TRACK
figs/*.png
figs/*.jpg
figs/*.pdf

# Python
__pycache__/
*.pyc
.venv/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Keep directory structure
!data/.gitkeep
!results/.gitkeep
!figs/.gitkeep
EOF

# 5. Xóa các file fix tạm khỏi repo (giữ lại local)
echo "[4] Dọn dẹp file tạm..."
rm -f fix_critical_v2.py
rm -f ikeda_fix.py
rm -f 池田先生_FIX.py
rm -f fix_data_final.py

# 6. Cập nhật README
echo "[5] Cập nhật README..."
cat > README.md << 'EOF'
# Credit Fairness Analysis - 池田Lab Workshop 2026

## Quick Start

```bash
# Setup
conda create -n credit python=3.11.9
conda activate credit
pip install -r requirements.txt

# Week 1: Baseline
python scripts/step3_baseline_ikeda.py

# Verify
python scripts/verify_week1_FINAL.py
```

## Expected Results
- Test Accuracy: ~0.75 (realistic, NOT 1.0)
- Data leakage: FIXED (no 'class' column)

## Structure
```
├── scripts/      # Python scripts
├── data/         # Data (not tracked)
├── results/      # Output (not tracked)
├── figs/         # Figures (not tracked)
└── docs/         # Documentation
```
EOF

echo "[6] Git operations..."
git add -A
git status

echo ""
echo "=========================================="
echo "HOÀN TẤT!"
echo "=========================================="
echo ""
echo "Kiểm tra git status ở trên, sau đó chạy:"
echo "  git commit -m '池田先生 FIX: Complete setup'"
echo "  git push origin main"
