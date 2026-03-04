# 与信スコアリングにおける公平性分析

## プロジェクト概要

機械学習モデルにおける予測精度と公平性のトレードオフを実証的に分析。  
German Credit Datasetを用いて、年齢・性別バイアスを定量化し、SHAP解析によりバイアス要因を特定。

## クイックスタート

### 1. 環境構築
```bash
pip install -r requirement.txt
```

### 2. 実験実行
```bash
# Week 1: ベースライン構築
python scripts/step1_download_data_FIXED.py
python scripts/step3_baseline_FIXED.py

# Week 2: 3モデル比較 (5-fold CV)
python scripts/run_experiment.py

# Week 3: SHAP解析
python scripts/week3_shap_analysis.py
```

### 3. 出力先
- **実験結果CSV**: `results/`
- **図表PNG**: `figs/`
- **報告書**: `docs/`

## 実験結果サマリー

### Week 1: ベースライン (Logistic Regression)
- Test Accuracy: **77.7%**
- AUC: **79.8%**
- DP_Age: 1.54% ✅
- DP_Sex: 4.17% ✅

### Week 2: モデル比較 (5-fold CV)
| モデル | Accuracy | DP_Age | DP_Sex |
|--------|----------|--------|--------|
| LR | 76.3±2.6% | 5.0±4.7% | 6.1±4.0% |
| RF | 75.9±1.2% | 5.9±4.3% | 3.1±2.5% |
| **XGBoost** | **77.8±2.5%** | 6.3±5.7% | **3.0±3.8%** |

### Week 3: SHAP解析 (バイアス要因特定)
1. **checking_status** (当座預金残高): 0.791
2. **credit_amount** (借入金額): 0.513
3. **duration** (返済期間): 0.394
4. **savings_status** (貯蓄残高): 0.364
5. **age** (年齢): 0.260 ← 保護属性の代理変数

## 主要な発見

✅ **精度と公平性の両立**: XGBoostが最高精度(77.8%)かつ最良性別公平性(DP_Sex: 3.0%)を達成  
✅ **Tree-basedモデルの優位性**: 非線形モデルが性別バイアスを約半減(LR: 6.1% → XGB: 3.0%)  
✅ **バイアスはデータに内在**: checking_statusとcredit_amountが主因(社会構造的格差)  
✅ **CV必須**: 単発評価は楽観的すぎる(Week1: 1.54% → Week2平均: 6.3%)

## リポジトリ構成
```
nguyen-credit-fairness-2026/
├── scripts/                    # 実行スクリプト
│   ├── step1_download_data_FIXED.py
│   ├── step3_baseline_FIXED.py
│   ├── run_experiment.py
│   └── week3_shap_analysis.py
├── results/                    # 実験結果CSV
│   ├── baseline_results_corrected.csv
│   ├── exp_v1_summary.csv
│   ├── exp_v1_all_folds.csv
│   └── shap_feature_importance.csv
├── docs/                       # 報告書・資料
│   ├── bias_hypothesis_report.md
│   ├── conclusion_limitation_future.md
│   └── presentation_slides_jp.md
├── README.md                   # このファイル
└── requirement.txt             # 依存ライブラリ
```

## 技術スタック

- Python 3.8+
- scikit-learn 1.3.0
- XGBoost 2.0.0
- SHAP 0.44.0
- Pandas, NumPy, Matplotlib, Seaborn

## 連絡先

**Nguyen Hoang**  
大和大学 理工学部 情報科学科  
学籍番号: 23610252  
GitHub: [@23610252hoang](https://github.com/23610252hoang)