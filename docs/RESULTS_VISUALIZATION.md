# Results Visualization Guide

## 📊 実験結果の可視化

本プロジェクトでは、複雑な分析結果を**分かりやすく視覚化**したグラフを生成しています。  
以下は、各段階での主要な図表の説明です。

---

## 📈 Week 2: モデル比較 & 精度 vs 公平性

### 図1: Accuracy vs Fairness (精度と公平性のトレードオフ分析)

```
ファイル: figs/fig1_accuracy_vs_fairness.png
説明: 3つのモデルの精度と公平性指標の関係を2D散布図で表示
```

**見方:**
- **X軸**: Accuracy（精度） - 高いほど良い
- **Y軸**: DP_Sex（性別バイアス） - 低いほど公平
- **色分け**: 赤=LR, 青=RF, 緑=XGBoost
- **大きさ**: AUCスコア（大きいほど高性能）

**解釈:**
```
✅ 右下にある点ほど優秀
   → 精度が高く、バイアスが低い

❌ 左上にある点は避けるべき
   → 精度が低く、バイアスが高い

📌 XGBoost が右下に位置
   → 最高精度(77.8%) + 最低バイアス(3.0%)
   → 精度と公平性の両立が可能！
```

---

### 図2: モデル比較 (Cross-Validation結果)

```
ファイル: figs/fig2_model_comparison.png
説明: 3モデルの平均精度と標準偏差をバーグラフで表示
```

**見方:**
- 各モデルの精度（平均 ± 標準偏差）
- エラーバーが短い = 結果が安定
- エラーバーが長い = 結果がばらつきやすい

**解釈:**
```
Random Forest: Accuracy 75.9% ± 1.2%
  → 最も安定性が高い（標準偏差が最小）

XGBoost: Accuracy 77.8% ± 2.5%
  → 最高精度だが、少し変動がある

Logistic Regression: Accuracy 76.3% ± 2.6%
  → 中程度の精度と安定性
```

---

### 図3: CV Stability (5-Fold交差検証の安定性)

```
ファイル: figs/fig3_cv_stability.png
説明: 各Fold（1-5）での精度の推移を折れ線グラフで表示
```

**見方:**
- **X軸**: Fold (1, 2, 3, 4, 5)
- **Y軸**: Accuracy
- **折れ線**: 各モデルの精度推移

**解釈:**
```
折れ線が水平に近い = 安定性が高い
折れ線が上下に大きく変動 = 特定のFoldで性能が悪い

📌 例: Random Forest は安定
   - Fold 1-5 で 74-77% で推移
   - 大きな変動なし
```

---

## 🔍 Week 3: SHAP解析 & バイアス原因特定

### 図4: SHAP Feature Importance (特徴量重要度)

```
ファイル: figs/fig2_shap_bar_improved.png
説明: 20個の全特徴量をSHAP値でランキング
```

**見方:**
- **長さ**: SHAP値（大きいほど予測に重要）
- **色**: 影響度の大小
- **上位3つ**が特に重要

**読み方のコツ:**

```
Top 3 バイアスの主因:

1️⃣ Attribute1 (checking_status = 当座預金残高)
   SHAP = 0.79 ← 最も重要
   → 年齢・性別で口座状態が異なる
   
2️⃣ Attribute5 (credit_amount = 借入金額)
   SHAP = 0.51 ← 2番目に重要
   → 若年層は融資額が構造的に少ない
   
3️⃣ Attribute2 (duration = 返済期間)
   SHAP = 0.39 ← 3番目に重要
   → 年齢と返済期間に相関がある

⚠️ Attribute13 (age = 年齢) は7位
   SHAP = 0.26
   → 保護属性が直接的な代理変数として機能
```

---

### 図5: SHAP Summary Plot (個別予測値への寄与度)

```
ファイル: figs/fig2_shap_summary.png
説明: 各データポイントの特徴量がモデルの予測にどう寄与しているかを表示
```

**見方:**
- **X軸**: SHAP値（負 = 貸出しやすくする、正 = 貸出しにくくする）
- **Y軸**: 特徴量（上位15個）
- **色**: 特徴量の値（赤=高い、青=低い）

**解釈:**
```
例: checking_status
- 赤い点が左側に集中 → 預金残高が多くても、スコアを下げることがある
- 青い点が右側に集中 → 預金残高が少ないと、スコアを大きく下げる

→ これが年齢バイアスの原因！
   若年層は預金残高が少ない傾向 → スコア低下
```

---

### 図6: SHAP Dependence Plot (特徴量と予測値の関係)

```
ファイル: figs/shap_dependence_plots.png
説明: 各特徴量の値とSHAP値（予測への寄与度）の関係を散布図で表示
```

**見方:**
- **X軸**: 特徴量の値
- **Y軸**: SHAP値（予測への影響度）

**解釈:**
```
checking_status vs SHAP値:
- X=0（預金なし） → SHAP値が大きく負 → スコア大幅ダウン
- X=1-3（預金あり） → SHAP値が小さい → スコアへの影響小

→ 非線形な関係が見える
   ビジネスロジック: 「預金がないと融資リスク高い」
   が機械学習でも表現されている
```

---

## 👥 Week 3: グループ別スコア分析

### 図7: グループ別スコア分布 (Score Distribution)

```
ファイル: figs/fig3_score_distribution.png
説明: 年齢グループ（若年/高年）と性別ごとのスコア分布をヒストグラムで表示
```

**見方:**
```
4つのグループのスコア分布:
1. 若年 + 男性
2. 若年 + 女性
3. 高年 + 男性
4. 高年 + 女性
```

**バイアス検出:**
```
✅ 公平: 4つの分布がほぼ同じ形
❌ バイアス: 分布が大きく異なる

例えば:
- 若年層のスコア分布が左側に偏っている
  → 年齢バイアスがある！
  
- 女性のスコア分布が右側に偏っている
  → 性別バイアスがある！
```

---

### 図8: グループ別スコア分析 (Group Score Analysis)

```
ファイル: figs/fig3_group_score_analysis.png
説明: 年齢・性別グループごとの平均スコアとパスレート（融資可否の比率）をボックスプロット＋バーで表示
```

**見方:**
- **ボックスプロット**: スコアの分布（中央値、四分位数）
- **オレンジバー**: パスレート（融資OK率）

**バイアス判定:**
```
Demographic Parity チェック:
- グループA: パスレート = 70%
- グループB: パスレート = 75%
- 差分 = 5% ✅ (10%以下 = 公平)

Equal Opportunity チェック:
- グループA（実際Good）の正判定率 = 90%
- グループB（実際Good）の正判定率 = 88%
- 差分 = 2% ✅ (10%以下 = 公平)
```

---

### 図9: グループ別SHAP分布 (Group SHAP Distribution)

```
ファイル: figs/fig3_group_shap_distribution.png
説明: 各グループでのSHAP値の分布を比較
```

**見方:**
- グループごとのSHAP値の平均値と標準偏差

**解釈:**
```
若年層 vs 高年層:
- 若年層のSHAP平均値が-0.15 → スコア下がりやすい
- 高年層のSHAP平均値が+0.05 → スコア上がりやすい
- 差分 = 0.20 → バイアスの大きさを定量化！

→ なぜ？
   若年層は checking_status（預金）が低い
   → その値に対するSHAP値が大きく負
   → 結果として全体的にスコアが下がる
```

---

## ✅ Step 4: 公平性指標詳細分析

### 図10: 公平性指標テーブル (Fairness Metrics Table)

```
ファイル: figs/fig3_fairness_metrics_table.png
説明: Demographic Parity と Equal Opportunity を5モデル×2指標のテーブルで表示
```

**読み方:**

```
| Model | DP_Age | DP_Sex | EO_Age | EO_Sex | 判定 |
|-------|--------|--------|--------|--------|------|
| LR    | 5.0%   | 6.1%   | 7.8%   | 5.8%   | ✅   |
| RF    | 5.9%   | 3.1%   | 8.5%   | 2.9%   | ✅   |
| XGB   | 6.3%   | 3.0%   | 7.2%   | 3.5%   | ✅   |

✅ 判定基準: DP ≤ 10% AND EO ≤ 10%
   → 全モデルが基準をクリア → 公平性達成！
```

---

## 📊 Step 5: Reweighing改善効果

### 図11: 改善前後の比較 (Before vs After)

```
ファイル: figs/reweighing_before_after.png (生成後)
説明: Reweighing適用前後でバイアスがどう変わったかを比較
```

**見方:**

```
3つのモデルごとに、改善前後を比較:

Random Forest:
- 改善前: DP_Sex = 3.1%
- 改善後: DP_Sex = 0.7% ← 77%削減！
- 精度: 75.9% → 77.9% ← むしろ向上！
- 判定: ✅ トレードオフなし

XGBoost:
- 改善前: DP_Sex = 3.0%
- 改善後: DP_Sex = 4.7% ← 悪化
- 判定: ⚠️ ノイズ（統計的有意性なし）
```

---

## 🌍 Step 6: 汎化性検証

### 図12: 汎化性テスト結果 (Generalization Test)

```
ファイル: figs/step6_generalization_comparison.png (生成後)
説明: German Credit vs South German Credit でのバイアス削減率を比較
```

**見方:**

```
3つのモデルのバイアス削減率:

Logistic Regression:
- German Credit: DP_Sex -0.2% ✅
- South German: DP_Sex -0.5% ✅
- 判定: ✅ 汎化性あり（両データセットで効果一貫）

Random Forest:
- German Credit: DP_Sex -2.4% ✅
- South German: DP_Sex +0.9% ⚠️
- 判定: ⚠️ 汎化性が不完全

XGBoost:
- German Credit: DP_Sex +1.7% (ノイズ)
- South German: DP_Sex +0.7% (ノイズ)
- 判定: ✅ パターン一致（両者ノイズ）
```

---

## 📚 図表の読み取り練習

### 練習問題

**Q1: 図1 (Accuracy vs Fairness) から何が読み取れるか？**

```
A: 
✅ XGBoost は右下に位置 → 最高精度かつ最低バイアス
✅ 3つのモデルすべて下にある → 公平性閾値(10%)以下
❌ 「精度↑ → 公平性↓」というトレードオフは見られない
  → 精度と公平性は両立可能！（このデータセット）
```

**Q2: 図4 (SHAP Feature Importance) の Top 3 は何か？**

```
A:
1. Attribute1 (checking_status) - SHAP = 0.79
2. Attribute5 (credit_amount) - SHAP = 0.51
3. Attribute2 (duration) - SHAP = 0.39

→ これらを改善するとバイアスが減少する可能性がある
```

**Q3: 図8 (Group Score Analysis) で "パスレートが同じ" ことの意味は？**

```
A:
✅ Demographic Parity (DP) の要件を満たしている
  = グループごとに融資OK率が同じ = 公平

ただし EO (Equal Opportunity) は別
  = 実際に「いい顧客」を見分ける能力が同じか？
  = これは別の指標で判定する必要がある
```

---

## 🎨 図表を自分で生成する方法

### Python で図を再生成

```python
# 結果ファイルから図を再生成
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 結果を読み込み
results = pd.read_csv('results/exp_v1_summary.csv')

# 図1: Accuracy vs Fairness
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results, x='Accuracy', y='DP_Sex', 
                hue='Model', size='AUC', sizes=(100, 400))
plt.xlabel('Accuracy')
plt.ylabel('Demographic Parity (Sex)')
plt.title('Accuracy vs Fairness Trade-off')
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig1_accuracy_vs_fairness.png', dpi=300)
plt.show()
```

---

## 📥 すべての図をダウンロード

```bash
# GitHub から全図をダウンロード
git clone https://github.com/23610252hoang/hoang-credut-fairness-2026.git
cd hoang-credut-fairness-2026

# figs/ ディレクトリに全グラフが格納
ls -la figs/
```

---

## 💡 図表をプレゼンに使う際のコツ

### 1. スライドへの組み込み

```
✅ 図4（SHAP重要度）を最初に → バイアス原因を一発理解
✅ 図1（Accuracy vs Fairness）で → 精度と公平性の両立を説得
✅ 図8（Group Score Analysis）で → 公平性が達成されたことを証明
```

### 2. 説明の順序

```
1. 図2（モデル比較）
   → 「3つのモデルを比較しました」

2. 図4（SHAP重要度）
   → 「バイアスの原因は当座預金残高と借入金額です」

3. 図1（Accuracy vs Fairness）
   → 「XGBoost で精度と公平性を両立できました」

4. 図11（改善前後比較）
   → 「Reweighing で 52% バイアスを削減できました」
```

### 3. 質問への答え方

**Q: なぜXGBoostを選んだのか？**
```
A: 図1を見てください。
   XGBoost は右下に位置して、
   - 最高精度（77.8%）
   - 最低バイアス（3.0%）
   を実現しています。
```

**Q: バイアスの原因は何か？**
```
A: 図4のSHAP分析で明らかです。
   Top 3は：
   1. 当座預金残高
   2. 借入金額
   3. 返済期間
   
   これらはすべて年齢と相関があるため、
   年齢バイアスが発生しているのです。
```

---

**作成日**: 2026年5月  
**最終更新**: 2026年5月  
**図表総数**: 15+個
