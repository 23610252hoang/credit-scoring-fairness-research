# 与信スコアリングにおける公平性分析

**学生:** Hoang Nguyen (グエン・キム・ホアン)  
**学籍番号:** 23610252  
**指導教員:** 池田教授  
**大学:** 大和大学 情報学部  
**期間:** 2026年2月 〜 2026年5月

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Status](https://img.shields.io/badge/進捗-Step%206%20完了-brightgreen.svg)]()

---

## 研究概要

本研究では、機械学習モデルにおける**予測精度と公平性のトレードオフ**を分析します。  
German Credit Dataset を使用し、**年齢**と**性別**に関するバイアスを定量的に評価します。  
さらに、**Reweighing による公平性改善**（事前処理による緩和手法）と、  
**South German Credit Dataset による汎化性の検証**まで実施しています。

### 研究課題（RQ）

> 「与信スコアリングモデルにおいて、精度を維持しながらバイアスを緩和できるか？  
>  また、その効果はデータセットをまたいで汎化するか？」

---

## 研究目的と進捗

| # | 目的 | 状況 |
|---|------|------|
| 1 | ベースラインモデル構築（LR, RF, XGBoost）+ データリーク修正 | ✅ 完了 |
| 2 | 公平性指標（DP, EO）の評価・5-fold CV による安定性検証 | ✅ 完了 |
| 3 | SHAP 値によるバイアス原因の特定と解釈 | ✅ 完了 |
| 4 | Reweighing による緩和手法の実装と効果検証 | ✅ 完了 |
| 5 | XGBoost の混在結果に関する相互作用分析 | ✅ 完了 |
| 6 | South German Credit による汎化性の検証 | ✅ 完了 |

---

## データセット

| 項目 | German Credit | South German Credit |
|------|--------------|---------------------|
| **ソース** | Hofmann (1994), UCI | Groemping (2019), UCI |
| **サンプル数** | 1,000 | 1,000 |
| **特徴量数** | 20 | 18 |
| **クラス分布** | Good 70% / Bad 30% | Good 70% / Bad 30% |
| **保護属性** | 年齢（age_binary: ≥40 = Old）, 性別（sex_binary） | 同左 |
| **備考** | 広く使用される標準データセット | famges=2 のコーディングに注意 |

---

## クイックスタート

### 環境構築

```bash
git clone https://github.com/23610252hoang/hoang-credut-fairness-2026.git
cd hoang-credut-fairness-2026

# Python 3.11 推奨
conda create -n credit python=3.11
conda activate credit
pip install -r requirements.txt
```

### 実行順序

```bash
# Step 1-3: ベースライン（データ前処理 → モデル比較 → SHAP）
python scripts/fix_data.py
python scripts/step3_baseline_FIXED_v2.py
python scripts/run_experiment.py
python scripts/week3_shap_analysis.py

# Step 4: 公平性分析
python scripts/step4_fairness_analysis.py

# Step 5: Reweighing 緩和手法
python scripts/step5_reweighing_mitigation.py

# Step 5b: XGBoost 混在結果の相互作用分析
python scripts/step5b_attribute_interaction_analysis.py

# Step 6: South German Credit による汎化性検証
python scripts/step6_south_german_credit.py
```

---

## 実験結果

### Step 1-2: ベースライン + モデル比較（5-Fold CV）

> 公平性閾値: **DP ≤ 10%**, **EO ≤ 10%**

| モデル | Accuracy | AUC | DP\_Age | EO\_Age | DP\_Sex | EO\_Sex |
|--------|----------|-----|---------|---------|---------|---------|
| Logistic Regression | 76.3±2.6% | 78.3±2.2% | 5.0±4.7% ✅ | 7.8±3.3% ✅ | 6.1±4.0% ✅ | 5.8±3.9% ✅ |
| Random Forest | 75.9±1.2% | 79.1±2.9% | 5.9±4.3% ✅ | 8.5±2.0% ✅ | 3.1±2.5% ✅ | 2.9±2.2% ✅ |
| **XGBoost** | **77.8±2.5%** | 78.4±3.1% | 6.3±5.7% ✅ | 7.2±2.9% ✅ | **3.0±3.8%** ✅ | 3.5±3.4% ✅ |

**主な発見:** 本データセット・本条件下では、精度と公平性のトレードオフは観察されなかった。XGBoost が最高精度かつ最低の性別バイアスを達成。

---

### Step 3: SHAP による特徴量重要度

| 順位 | 特徴量 | 英語名 | SHAP 値 | バイアスとの関連 |
|------|--------|--------|---------|----------------|
| 1 | Attribute1 | checking\_status | 0.791 | 年齢・性別間で口座状態に差 |
| 2 | Attribute5 | credit\_amount | 0.513 | 若年層は融資額が構造的に少ない |
| 3 | Attribute2 | duration | 0.394 | 年齢と返済期間に相関あり |
| **7** | **Attribute13** | **age** | **0.260** | **保護属性の直接代理変数** |
| 14 | Attribute9 | personal\_status | 0.097 | 性別情報を直接含む |

**主な発見:** バイアスはモデルではなく、社会的不平等を反映したデータ構造に内在する。

---

### Step 5: Reweighing 緩和手法（Kamiran & Calders, 2012）

Reweighing はサンプルに重みを付与することで保護属性とラベルの統計的依存を除去する事前処理手法。追加ライブラリ不要、スクラッチ実装。

**German Credit Dataset での結果（Δ = Reweighing − Baseline）:**

| モデル | Acc Δ | DP\_sex Δ | EO\_sex Δ | DP\_age Δ | EO\_age Δ |
|--------|-------|-----------|-----------|-----------|-----------|
| Logistic Regression | −0.001 | −0.002 | −0.003 | −0.001 | −0.002 |
| **Random Forest** | **+0.020** | **−0.024** | **−0.013** | −0.011 | **−0.030** |
| XGBoost | −0.001 | +0.017 ⚠️ | +0.016 ⚠️ | −0.011 | −0.013 |

- **Random Forest**: DP\_sex が 52% 減少（0.046→0.022）、精度も +2% 向上 — トレードオフなし
- **XGBoost**: 性別公平性がわずかに悪化（+0.017）、年齢公平性は改善

---

### Step 5b: XGBoost 混在結果の原因分析

XGBoost の DP\_sex 悪化（+0.017）を 3 仮説で検証:

| 仮説 | 内容 | 結果 |
|------|------|------|
| E1: 重みシフトによる age 分布の歪み | Reweighing が age グループの有効サンプル数を変化させる | **否定**: 変化量 < 0.3%（ほぼ無影響） |
| **E2: 高分散ノイズ** | N=1000 での公平性推定値は高分散 | **支持**: std が重複（DP\_sex baseline: 0.039±0.030, reweighing: 0.056±0.025）→ 統計的に有意でない |
| E3: 特徴量相互作用カップリング | XGBoost の木構造が sex/age を間接的に結合 | **否定**: 最大 delta = 0.008（微小） |

**結論:** XGBoost の DP\_sex 悪化は統計的ノイズであり、実質的なトレードオフを示すものではない。

---

### Step 6: South German Credit による汎化性検証

**クロスデータセット比較（ΔDP\_sex の方向一致性）:**

| モデル | German Credit | South German | 汎化? |
|--------|--------------|--------------|-------|
| Logistic Regression | −0.002 ✅ | −0.005 ✅ | ✅ 一致 |
| Random Forest | −0.024 ✅ | +0.009 ⚠️ | ⚠️ 不一致 |
| XGBoost | +0.017 (noise) | +0.007 (noise) | ✅ パターン一致 |

- **LR**: 両データセットで一貫した改善 → 結論が汎化する
- **RF**: German Credit の劇的改善が South German では再現されない  
  → 原因: famges=2 の曖昧コーディング（Female/Male 混在）、グループ比率の差異（Female: 57% vs 40%）
- **XGBoost**: 両データセットで混在パターン → Step 5b の解釈が汎化

---

## ディレクトリ構造

```
hoang-credut-fairness-2026/
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── fix_data.py                           # データ前処理
│   ├── step3_baseline_FIXED_v2.py            # Step 1: LR ベースライン
│   ├── step4_fairness_analysis.py            # Step 2: 公平性分析
│   ├── run_experiment.py                     # Step 2: 3 モデル + 5-fold CV
│   ├── week3_shap_analysis.py                # Step 3: SHAP 解析
│   ├── step5_reweighing_mitigation.py        # Step 5: Reweighing 緩和手法 ★
│   ├── step5b_attribute_interaction_analysis.py  # Step 5b: 相互作用分析 ★
│   └── step6_south_german_credit.py          # Step 6: 汎化性検証 ★
│
├── data/
│   ├── german_credit_processed.csv           # German Credit（前処理済み）
│   └── south_german_credit_processed.csv     # South German Credit（Step 6 生成）
│
├── results/
│   ├── exp_v1_summary.csv                    # Step 2: モデル比較サマリー
│   ├── exp_v1_all_folds.csv                  # Step 2: fold 別詳細
│   ├── shap_feature_importance.csv           # Step 3: SHAP 重要度
│   ├── mitigation_summary.csv                # Step 5: Reweighing サマリー
│   ├── step5b_fold_comparison.csv            # Step 5b: fold 別 DP 比較
│   ├── step6_south_german_summary.csv        # Step 6: South German サマリー
│   └── step6_cross_dataset_comparison.csv    # Step 6: クロスデータセット比較
│
├── figs/
│   ├── fig2_model_comparison.png             # Step 2: モデル比較
│   ├── fig2_shap_bar_improved.png            # Step 3: SHAP 重要度
│   ├── fig_mitigation_comparison.png         # Step 5: Reweighing 効果
│   ├── fig5b_interaction_analysis.png        # Step 5b: 相互作用分析
│   └── fig6_cross_dataset_comparison.png     # Step 6: クロスデータセット
│
├── docs/
│   ├── weekly_report_week1.md                # Week 1 進捗報告
│   └── bias_hypothesis_report.md             # バイアス仮説レポート
│
└── poster/
    ├── 23610252_NGUYENKIMHOANG 0329 ポスター.pptx
    └── poster_FINAL.pdf
```

---

## 技術仕様

### モデル設定

```python
LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')

RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
```

### 評価指標

```
Demographic Parity (DP):    |P(Ŷ=1|A=0) − P(Ŷ=1|A=1)|
Equal Opportunity  (EO):    |TPR(A=0)   − TPR(A=1)  |
公平性閾値:                  DP ≤ 10%, EO ≤ 10%
```

### Reweighing の重み計算

```
w(xᵢ) = P(Y=yᵢ) × P(A=aᵢ) / P(Y=yᵢ, A=aᵢ)
```

---

## 進捗状況

| Step | テーマ | 状況 | 主な成果物 |
|------|--------|------|-----------|
| **Step 1** | データリーク修正・ベースライン | ✅ 完了 | `baseline_results.csv` |
| **Step 2** | 3 モデル比較・5-fold CV | ✅ 完了 | `exp_v1_summary.csv` |
| **Step 3** | SHAP・バイアス特定 | ✅ 完了 | `shap_feature_importance.csv` |
| **Step 4** | 公平性指標詳細分析 | ✅ 完了 | `fairness_metrics.csv` |
| **Step 5** | Reweighing 緩和手法 | ✅ 完了 | `mitigation_summary.csv` |
| **Step 5b** | XGBoost 相互作用分析 | ✅ 完了 | `step5b_*.csv` |
| **Step 6** | South German 汎化性検証 | ✅ 完了 | `step6_*.csv` |

---

## 参考文献

1. Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning.* NeurIPS.
2. Kamiran, F., & Calders, T. (2012). *Data preprocessing techniques for classification without discrimination.* KAIS.
3. Groemping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set.* Beuth University, Report 4/2019.
4. Lundberg, S., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
5. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning.* fairmlbook.org.

---

## 連絡先

**グエン・キム・ホアン（Hoang Nguyen）**  
大和大学 情報学部 4 年  
指導教員: 池田教授  
Email: 23610252kn@stu.yamato-u.ac.jp  
GitHub: [@23610252hoang](https://github.com/23610252hoang)

---

**最終更新:** 2026年5月  
**ステータス:** Step 6 完了 ✅ | 論文執筆中
