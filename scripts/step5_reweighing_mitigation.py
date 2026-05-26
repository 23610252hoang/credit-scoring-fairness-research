#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5: Reweighing Mitigation — Before/After Fairness Comparison
Pre-processing bias mitigation using the Reweighing algorithm
(Kamiran & Calders, 2012). No extra libraries required.

Usage (from project root):
    python scripts/step5_reweighing_mitigation.py

Output:
    results/mitigation_results.csv   — fold-level detail
    results/mitigation_summary.csv   — mean ± std per condition
    figs/fig_mitigation_comparison.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
N_SPLITS = 5
PROTECTED_ATTRS = ['sex_binary', 'age_binary']
EXCLUDE_COLS = ['target', 'age_binary', 'sex_binary', 'age_group', 'sex_group', 'class']

# ── Reweighing Algorithm ─────────────────────────────────────────────────────

def compute_reweighing_weights(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    """
    Kamiran & Calders (2012) Reweighing.

    Each sample receives weight:
        w = P(Y=y) * P(A=a) / P(Y=y, A=a)

    This makes the joint distribution of (Y, A) appear independent,
    removing the statistical association between the label and the
    protected attribute without altering the dataset itself.
    """
    n = len(y)
    weights = np.ones(n)

    for a_val in np.unique(sensitive):
        for y_val in np.unique(y):
            mask = (sensitive == a_val) & (y == y_val)
            if mask.sum() == 0:
                continue
            p_y = (y == y_val).mean()
            p_a = (sensitive == a_val).mean()
            p_ya = mask.mean()
            weights[mask] = (p_y * p_a) / p_ya

    return weights

# ── Data Loading & Preprocessing ─────────────────────────────────────────────

def load_and_preprocess(path: str = 'data/german_credit_processed.csv'):
    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X_df = df[feature_cols].copy()

    for col in X_df.select_dtypes(include=['object']).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X = X_df.values
    y = df['target'].values
    sensitive = {attr: df[attr].values for attr in PROTECTED_ATTRS}

    return X, y, sensitive

# ── Fairness Metrics ──────────────────────────────────────────────────────────

def fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     sensitive: np.ndarray) -> dict:
    """
    Returns:
        dp_diff   — Demographic Parity difference  |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
        eo_diff   — Equal Opportunity difference   |TPR(A=0) - TPR(A=1)|
        eodds_diff— Equalized Odds: max(|ΔTPR|, |ΔFPR|)
    """
    results = {}
    for a_val in [0, 1]:
        mask = sensitive == a_val
        results[f'approve_{a_val}'] = y_pred[mask].mean()
        pos_mask = mask & (y_true == 1)
        neg_mask = mask & (y_true == 0)
        tp = ((y_pred == 1) & pos_mask).sum()
        fn = ((y_pred == 0) & pos_mask).sum()
        fp = ((y_pred == 1) & neg_mask).sum()
        tn = ((y_pred == 0) & neg_mask).sum()
        results[f'tpr_{a_val}'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[f'fpr_{a_val}'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    dp   = abs(results['approve_0'] - results['approve_1'])
    eo   = abs(results['tpr_0']     - results['tpr_1'])
    eodds = max(eo, abs(results['fpr_0'] - results['fpr_1']))

    return {'dp_diff': dp, 'eo_diff': eo, 'eodds_diff': eodds}

# ── Single Fold Evaluation ────────────────────────────────────────────────────

def eval_fold(model, X_tr, y_tr, X_te, y_te, sensitive_te,
              sample_weight=None) -> dict:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight

    model.fit(X_tr_s, y_tr, **fit_kwargs)

    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    row = {
        'accuracy': accuracy_score(y_te, y_pred),
        'auc':      roc_auc_score(y_te, y_prob),
    }
    for attr_name, s_te in sensitive_te.items():
        fm = fairness_metrics(y_te, y_pred, s_te)
        row[f'dp_{attr_name}']    = fm['dp_diff']
        row[f'eo_{attr_name}']    = fm['eo_diff']
        row[f'eodds_{attr_name}'] = fm['eodds_diff']

    return row

# ── Cross-Validation Runner ───────────────────────────────────────────────────

def run_cv(model_name: str, model, X: np.ndarray, y: np.ndarray,
           sensitive: dict, condition: str) -> pd.DataFrame:
    """
    condition: 'Baseline' or 'Reweighing'
    For Reweighing, weights are computed on the training fold only
    (no leakage into the test fold).
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        s_te = {k: v[te_idx] for k, v in sensitive.items()}

        weights = None
        if condition == 'Reweighing':
            # Reweigh on sex_binary (primary protected attribute)
            weights = compute_reweighing_weights(y_tr, sensitive['sex_binary'][tr_idx])

        row = eval_fold(model, X_tr, y_tr, X_te, y_te, s_te,
                        sample_weight=weights)
        row.update({'model': model_name, 'condition': condition, 'fold': fold})
        rows.append(row)
        print(f"  [{condition}] {model_name} fold {fold}: "
              f"Acc={row['accuracy']:.4f} "
              f"DP_sex={row['dp_sex_binary']:.4f} "
              f"EO_sex={row['eo_sex_binary']:.4f}")

    return pd.DataFrame(rows)

# ── Summary Table ─────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns
                   if c not in ('model', 'condition', 'fold')]
    agg = df.groupby(['model', 'condition'])[metric_cols].agg(['mean', 'std'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    return agg.reset_index()

# ── Comparison Plot ───────────────────────────────────────────────────────────

def make_comparison_plot(summary: pd.DataFrame, out_path: str):
    models = summary['model'].unique()
    metrics = [
        ('accuracy_mean',    'accuracy_std',    'Accuracy'),
        ('auc_mean',         'auc_std',         'AUC'),
        ('dp_sex_binary_mean',  'dp_sex_binary_std',  'DP (Sex)'),
        ('eo_sex_binary_mean',  'eo_sex_binary_std',  'EO (Sex)'),
        ('dp_age_binary_mean',  'dp_age_binary_std',  'DP (Age)'),
        ('eo_age_binary_mean',  'eo_age_binary_std',  'EO (Age)'),
    ]

    n_metrics = len(metrics)
    n_models  = len(models)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4.5))
    fig.suptitle('Baseline vs Reweighing Mitigation\n(5-Fold CV, German Credit Dataset)',
                 fontsize=13, fontweight='bold', y=1.02)

    colors = {'Baseline': '#4C72B0', 'Reweighing': '#DD8452'}
    bar_w  = 0.35
    x      = np.arange(n_models)

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        for i, cond in enumerate(['Baseline', 'Reweighing']):
            sub = summary[summary['condition'] == cond].set_index('model')
            means = [sub.loc[m, mean_col] if m in sub.index else 0 for m in models]
            stds  = [sub.loc[m, std_col]  if m in sub.index else 0 for m in models]
            offset = (i - 0.5) * bar_w
            ax.bar(x + offset, means, bar_w, yerr=stds, capsize=4,
                   color=colors[cond], alpha=0.88, label=cond,
                   error_kw={'elinewidth': 1.2})

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    handles = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.0), fontsize=10, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ── Delta Table (for thesis) ──────────────────────────────────────────────────

def print_delta_table(summary: pd.DataFrame):
    cols_of_interest = [
        ('accuracy_mean', 'accuracy_std', 'Accuracy'),
        ('auc_mean',      'auc_std',      'AUC'),
        ('dp_sex_binary_mean', 'dp_sex_binary_std', 'DP_sex'),
        ('eo_sex_binary_mean', 'eo_sex_binary_std', 'EO_sex'),
        ('dp_age_binary_mean', 'dp_age_binary_std', 'DP_age'),
        ('eo_age_binary_mean', 'eo_age_binary_std', 'EO_age'),
    ]

    print("\n" + "=" * 80)
    print("THESIS TABLE — Before vs After Reweighing (mean ± std, 5-Fold CV)")
    print("=" * 80)

    header = f"{'Model':<22} {'Condition':<12}"
    for _, _, label in cols_of_interest:
        header += f" {label:>14}"
    print(header)
    print("-" * 80)

    for model in summary['model'].unique():
        for cond in ['Baseline', 'Reweighing']:
            row = summary[(summary['model'] == model) &
                          (summary['condition'] == cond)]
            if row.empty:
                continue
            row = row.iloc[0]
            line = f"{model:<22} {cond:<12}"
            for mean_col, std_col, _ in cols_of_interest:
                val = f"{row[mean_col]:.4f}±{row[std_col]:.4f}"
                line += f" {val:>14}"
            print(line)
        print()

    # Delta rows (Baseline - Reweighing, positive = improvement for fairness metrics)
    print("-" * 80)
    print("Δ = Reweighing − Baseline  (negative Δ on fairness = IMPROVEMENT)")
    print("-" * 80)
    for model in summary['model'].unique():
        base = summary[(summary['model'] == model) &
                       (summary['condition'] == 'Baseline')]
        rw   = summary[(summary['model'] == model) &
                       (summary['condition'] == 'Reweighing')]
        if base.empty or rw.empty:
            continue
        base, rw = base.iloc[0], rw.iloc[0]
        line = f"{model:<22} {'Δ':<12}"
        for mean_col, _, _ in cols_of_interest:
            delta = rw[mean_col] - base[mean_col]
            sign  = '+' if delta >= 0 else ''
            line += f" {sign + f'{delta:.4f}':>14}"
        print(line)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 5: Reweighing Mitigation")
    print("Kamiran & Calders (2012)")
    print("=" * 60)

    print("\n[1] Loading data...")
    X, y, sensitive = load_and_preprocess()
    print(f"    Samples: {len(y)}  |  Features: {X.shape[1]}")
    print(f"    sex_binary  Female: {(sensitive['sex_binary']==0).sum()}, "
          f"Male: {(sensitive['sex_binary']==1).sum()}")
    print(f"    age_binary  Young:  {(sensitive['age_binary']==0).sum()}, "
          f"Old:   {(sensitive['age_binary']==1).sum()}")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs'),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric='logloss',
            use_label_encoder=False),
    }

    print("\n[2] Running 5-Fold CV (Baseline + Reweighing) ...")
    all_results = []
    for model_name, model in models.items():
        print(f"\n  -- {model_name} --")
        for condition in ['Baseline', 'Reweighing']:
            df_fold = run_cv(model_name, model, X, y, sensitive, condition)
            all_results.append(df_fold)

    results_df = pd.concat(all_results, ignore_index=True)

    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/mitigation_results.csv', index=False)
    print("\nSaved: results/mitigation_results.csv")

    summary = summarise(results_df)
    summary.to_csv('results/mitigation_summary.csv', index=False)
    print("Saved: results/mitigation_summary.csv")

    print_delta_table(summary)

    print("\n[3] Creating comparison plot ...")
    make_comparison_plot(summary, 'figs/fig_mitigation_comparison.png')

    print("\n" + "=" * 60)
    print("Step 5 complete.")
    print("Next step: python scripts/step6_generalization.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
