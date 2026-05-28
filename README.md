# 与信スコアリングにおける公平性分析

**Student:** Hoang Nguyen（グエン・キム・ホアン）  
**Student ID:** 23610252  
**University:** 大和大学 情報学部  
**Advisor:** 池田教授  
**Period:** 2026年2月 - 2026年5月

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-brightgreen.svg)

## 概要

本プロジェクトは、与信スコアリングモデルにおいて、予測精度だけでなく公平性と説明可能性も評価するための分析プロトタイプです。

金融機関・保険会社・採用支援システムなどでAIを意思決定支援に使う場合、モデルの性能だけでなく、特定の属性に対する不利な影響、判断根拠の説明、運用時のリスク管理が重要になります。本研究では、German Credit Datasetを題材に、複数の機械学習モデルを比較し、属性別の公平性指標とSHAPによる説明可能性を分析しました。

## 研究課題

本プロジェクトでは、次の問いを扱います。

1. 与信スコアリングモデルで、精度と公平性をどのように同時評価できるか
2. 年齢・性別グループごとの予測差をどのように定量化できるか
3. SHAPを用いて、モデル判断の主要因をどのように説明できるか
4. 公平性改善手法を、実務上のモデル監査プロセスにどう接続できるか

## 主な成果

| 観点 | 内容 |
|---|---|
| モデル比較 | Logistic Regression、Random Forest、XGBoostを比較 |
| 性能評価 | Accuracy、AUCを用いて予測性能を評価 |
| 公平性評価 | Demographic Parity、Equal Opportunityを用いて属性別の差を評価 |
| 説明可能性 | SHAPにより特徴量の寄与度を可視化 |
| 実装上の学び | データリークを修正し、現実的なベースラインを再構築 |

初期実装では目的変数が特徴量に混入するデータリークにより、過度に高いAccuracyが出る問題がありました。これを修正した後、ベースラインモデルとしてAccuracy約72.5%、AUC約0.756を確認しました。この経験は、実務におけるモデル検証とデータ品質確認の重要性を示しています。

## 技術スタック

- Python 3.11
- pandas / NumPy
- scikit-learn
- XGBoost
- SHAP
- matplotlib / seaborn

## セットアップ

```bash
git clone https://github.com/23610252hoang/hoang-credut-fairness-2026.git
cd hoang-credut-fairness-2026

conda create -n credit python=3.11
conda activate credit
pip install -r requirements.txt
```

## 実行例

```bash
python scripts/fix_data.py
python scripts/step3_baseline_FIXED_v2.py
python scripts/run_experiment.py
python scripts/week3_shap_analysis.py
python scripts/step5_reweighing_mitigation.py
python scripts/step6_south_german_credit.py
```

## 主なファイル

| パス | 内容 |
|---|---|
| `scripts/step3_baseline_FIXED_v2.py` | データリーク修正後のベースラインモデル |
| `scripts/run_experiment.py` | 複数モデルの5-Fold Cross-Validation |
| `scripts/week3_shap_analysis.py` | SHAPによる説明可能性分析 |
| `scripts/step5_reweighing_mitigation.py` | Reweighingによる公平性改善の検証 |
| `scripts/step6_south_german_credit.py` | South German Credit Datasetでの追加検証 |
| `results/exp_v1_summary.csv` | モデル比較結果 |
| `results/shap_feature_importance.csv` | SHAP特徴量重要度 |
| `figs/fig1_accuracy_vs_fairness.png` | 精度と公平性の比較図 |
| `figs/fig2_shap_summary.png` | SHAP summary plot |
| `docs/企業向け要約.md` | 採用担当者・企業向けの要約 |
| `docs/RESULTS_VISUALIZATION.md` | 図表の読み方と説明 |
| `docs/企業面接対策.md` | 面接準備用メモ |
| `docs/interview_pitch_ja.md` | 企業面接向けの正式プレゼン原稿 |

## 実験結果の要約

### ベースライン

データリーク修正後のベースラインでは、以下の現実的な性能を確認しました。

| 指標 | 値 |
|---|---|
| Accuracy | 約72.5% |
| AUC | 約0.756 |

### モデル比較

`run_experiment.py`では、Logistic Regression、Random Forest、XGBoostを5-Fold Cross-Validationで比較します。性能指標に加え、年齢・性別グループごとのDemographic ParityとEqual Opportunityを評価します。

### 説明可能性分析

SHAPを用いて、モデル予測に影響する特徴量を分析しました。これにより、単に「どのモデルが高性能か」だけでなく、「なぜその判断になるのか」を説明しやすくなります。

## 企業での応用可能性

本プロジェクトは、次のような業務課題に応用できます。

- 与信審査モデルの導入前監査
- 属性別の不公平リスクの検出
- モデル判断の説明資料作成
- リスク管理部門・法務部門との合意形成
- モデル更新時の品質確認

実務では、AIモデルは単独で意思決定を完結させるものではなく、人間の判断を支援する仕組みとして使われることが多くあります。そのため、精度、公平性、説明可能性、運用監視を同時に確認することが重要です。

## 法規制・ガイドラインとの関係

本プロジェクトは法的助言を目的としたものではありませんが、AIガバナンスの観点で次の動向と関連します。

- EU GDPR Article 22は、法的効果または同等に重大な影響をもつ完全自動化された個人判断について、データ主体の権利を定めています。
- 日本では、総務省・経済産業省の「AI事業者ガイドライン」が、AIの公平性、透明性、アカウンタビリティ、リスク管理を重視しています。
- 企業でAIを運用する場合、個人情報保護、説明責任、社内ガバナンス、継続的なモニタリングを組み合わせて考える必要があります。

## 今後の改善案

- StreamlitまたはDashによるモデル監査ダッシュボード化
- 閾値変更による承認率・リスク・収益影響のシミュレーション
- モデル更新時の公平性指標チェックをCI/CDに組み込む
- 実務データを想定したデータドリフト監視
- 結果ファイルと図表の自動再生成手順の整備

## 参考文献・参考資料

- Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning.*
- Kamiran, F., & Calders, T. (2012). *Data Preprocessing Techniques for Classification without Discrimination.*
- Lundberg, S., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.*
- Groemping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set.*
- European Union. GDPR Article 22: Automated individual decision-making, including profiling.
- 総務省・経済産業省. AI事業者ガイドライン.

## 連絡先

**Hoang Nguyen（グエン・キム・ホアン）**

- GitHub: [@23610252hoang](https://github.com/23610252hoang)
- Repository: https://github.com/23610252hoang/hoang-credut-fairness-2026

---

**Status:** Research prototype  
**Last updated:** 2026年5月
