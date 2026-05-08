#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6: Generalization — South German Credit Dataset
Tests whether findings from German Credit hold on the South German Credit
dataset (Groemping 2019, N=1000), which corrects coding errors in the
original Hofmann (1994) dataset.

Research question: Are the Step 5 conclusions (Reweighing benefits RF
most; XGBoost mixed results are variance noise) specific to German Credit
or do they generalise to a closely related dataset?

Usage (from project root):
    python scripts/step6_south_german_credit.py

Outputs:
    data/south_german_credit_processed.csv
    results/step6_south_german_results.csv
    results/step6_south_german_summary.csv
    results/step6_cross_dataset_comparison.csv
    figs/fig6_cross_dataset_comparison.png

Reference:
    Groemping, U. (2019). South German Credit Data: Correcting a Widely
    Used Data Set. Beuth University of Applied Sciences Berlin,
    Report 4/2019.
"""

import os
import ssl
import zipfile
import io
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_SPLITS = 5

# Age threshold consistent with German Credit preprocessing (~29% Old)
AGE_THRESHOLD = 40

# famges values that map to Female (includes ambiguous category 2:
# "female non-single OR male single" -- acknowledged as limitation)
FEMALE_FAMGES = [2, 4]

COL_NAMES = [
    'status', 'duration', 'credit_history', 'purpose', 'amount',
    'savings', 'employment', 'installment_rate', 'personal_status_sex',
    'other_debtors', 'residence', 'property', 'age', 'other_plans',
    'housing', 'num_credits', 'job', 'people_liable', 'telephone',
    'foreign_worker', 'credit_risk'
]


# ── Download helper ───────────────────────────────────────────────────────────

def download_south_german(out_path: str = 'data/south_german_tmp/SouthGermanCredit.asc'):
    if os.path.exists(out_path):
        print(f"    Using cached: {out_path}")
        return out_path

    print("    Downloading South German Credit from UCI ...")
    url = 'https://archive.ics.uci.edu/static/public/573/south+german+credit+update.zip'
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(url, context=ctx, timeout=30) as r:
        data = r.read()

    z = zipfile.ZipFile(io.BytesIO(data))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    z.extractall(os.path.dirname(out_path))
    print(f"    Extracted: {z.namelist()}")
    return out_path


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_south_german(raw_path: str) -> pd.DataFrame:
    """
    Load and preprocess South German Credit.

    Protected attributes (kept separate, not used as features):
      sex_binary : 0=Female (famges in {2,4}), 1=Male (famges in {1,3})
                   NOTE: famges=2 is ambiguous (female non-single OR male
                   single); classified as Female following the convention
                   that most German Credit fairness studies use.
      age_binary : 0=Young (age < 40), 1=Old (age >= 40)
                   Threshold 40 matches the ~29% Old split in German Credit.
    """
    df = pd.read_csv(raw_path, sep=' ', header=0)
    df.columns = COL_NAMES

    # Target: 0=bad, 1=good (already binary in Groemping 2019)
    df['target'] = df['credit_risk']

    # Protected attributes
    df['sex_binary'] = df['personal_status_sex'].apply(
        lambda x: 0 if x in FEMALE_FAMGES else 1)
    df['age_binary'] = (df['age'] >= AGE_THRESHOLD).astype(int)

    # Feature columns: exclude raw protected attrs and target
    drop_cols = ['credit_risk', 'personal_status_sex', 'age']
    df_clean = df.drop(columns=drop_cols)

    return df_clean


def get_feature_matrix(df: pd.DataFrame):
    """Split into X (features), y, and protected attribute arrays."""
    exclude = ['target', 'sex_binary', 'age_binary']
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values.astype(float)
    y = df['target'].values
    sex = df['sex_binary'].values
    age = df['age_binary'].values
    return X, y, sex, age


# ── Reweighing ────────────────────────────────────────────────────────────────

def compute_reweighing_weights(y, sensitive):
    weights = np.ones(len(y))
    for a in np.unique(sensitive):
        for yv in np.unique(y):
            mask = (sensitive == a) & (y == yv)
            if mask.sum() == 0:
                continue
            weights[mask] = (y == yv).mean() * (sensitive == a).mean() / mask.mean()
    return weights


# ── Fairness metrics ──────────────────────────────────────────────────────────

def fairness_metrics(y_true, y_pred, sensitive):
    out = {}
    for a in [0, 1]:
        m = sensitive == a
        out[f'approve_{a}'] = y_pred[m].mean()
        pos = m & (y_true == 1)
        neg = m & (y_true == 0)
        tp = ((y_pred == 1) & pos).sum()
        fn = ((y_pred == 0) & pos).sum()
        fp = ((y_pred == 1) & neg).sum()
        tn = ((y_pred == 0) & neg).sum()
        out[f'tpr_{a}'] = tp / (tp + fn) if (tp + fn) else 0.0
        out[f'fpr_{a}'] = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        'dp_diff':    abs(out['approve_0'] - out['approve_1']),
        'eo_diff':    abs(out['tpr_0']     - out['tpr_1']),
        'eodds_diff': max(abs(out['tpr_0'] - out['tpr_1']),
                         abs(out['fpr_0'] - out['fpr_1'])),
    }


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_cv(model_name, model, X, y, sex, age, condition):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        w = compute_reweighing_weights(y_tr, sex[tr]) if condition == 'Reweighing' else None
        model.fit(X_tr_s, y_tr, **({'sample_weight': w} if w is not None else {}))

        y_pred = model.predict(X_te_s)
        y_prob = model.predict_proba(X_te_s)[:, 1]

        fm_sex = fairness_metrics(y_te, y_pred, sex[te])
        fm_age = fairness_metrics(y_te, y_pred, age[te])

        rows.append({
            'model': model_name, 'condition': condition, 'fold': fold,
            'accuracy': accuracy_score(y_te, y_pred),
            'auc':      roc_auc_score(y_te, y_prob),
            'dp_sex':   fm_sex['dp_diff'],   'eo_sex':  fm_sex['eo_diff'],
            'dp_age':   fm_age['dp_diff'],   'eo_age':  fm_age['eo_diff'],
        })
        print(f"    [{condition}] {model_name} fold {fold}: "
              f"Acc={rows[-1]['accuracy']:.4f}  "
              f"DP_sex={rows[-1]['dp_sex']:.4f}  "
              f"DP_age={rows[-1]['dp_age']:.4f}")
    return pd.DataFrame(rows)


def summarise(df):
    metric_cols = ['accuracy', 'auc', 'dp_sex', 'eo_sex', 'dp_age', 'eo_age']
    agg = df.groupby(['model', 'condition'])[metric_cols].agg(['mean', 'std'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    return agg.reset_index()


# ── Cross-dataset comparison table ────────────────────────────────────────────

def load_german_credit_summary():
    """Load Step 5 summary for German Credit and standardise column names."""
    path = 'results/mitigation_summary.csv'
    if not os.path.exists(path):
        print("  WARNING: results/mitigation_summary.csv not found.")
        print("  Run step5_reweighing_mitigation.py first.")
        return None

    raw = pd.read_csv(path)
    # Step 5 columns use 'sex_binary' / 'age_binary' suffix
    rename = {}
    for c in raw.columns:
        nc = c.replace('_sex_binary', '_sex').replace('_age_binary', '_age')
        rename[c] = nc
    return raw.rename(columns=rename)


def build_cross_dataset_table(german_sum, sg_sum):
    metrics = [
        ('accuracy_mean', 'accuracy_std', 'Accuracy'),
        ('auc_mean',      'auc_std',      'AUC'),
        ('dp_sex_mean',   'dp_sex_std',   'DP_sex'),
        ('eo_sex_mean',   'eo_sex_std',   'EO_sex'),
        ('dp_age_mean',   'dp_age_std',   'DP_age'),
        ('eo_age_mean',   'eo_age_std',   'EO_age'),
    ]
    rows = []
    for ds_name, df in [('German Credit', german_sum),
                        ('South German', sg_sum)]:
        for _, row in df.iterrows():
            r = {'dataset': ds_name,
                 'model': row['model'], 'condition': row['condition']}
            for mc, sc, label in metrics:
                r[label] = f"{row[mc]:.4f}+/-{row[sc]:.4f}"
                r[f'{label}_mean'] = row[mc]
                r[f'{label}_std']  = row[sc]
            rows.append(r)
    return pd.DataFrame(rows)


def print_cross_dataset_table(table):
    metrics = ['Accuracy', 'AUC', 'DP_sex', 'EO_sex', 'DP_age', 'EO_age']
    print("\n" + "=" * 90)
    print("CROSS-DATASET COMPARISON  (mean +/- std, 5-Fold CV)")
    print("=" * 90)
    header = f"{'Dataset':<16} {'Model':<22} {'Cond':<12}"
    for m in metrics:
        header += f" {m:>17}"
    print(header)
    print("-" * 90)
    for ds in ['German Credit', 'South German']:
        sub = table[table['dataset'] == ds]
        for model in sub['model'].unique():
            for cond in ['Baseline', 'Reweighing']:
                row = sub[(sub['model'] == model) & (sub['condition'] == cond)]
                if row.empty:
                    continue
                row = row.iloc[0]
                line = f"{ds:<16} {model:<22} {cond:<12}"
                for m in metrics:
                    line += f" {row[m]:>17}"
                print(line)
        print()


# ── Figure ────────────────────────────────────────────────────────────────────

def make_cross_dataset_figure(german_sum, sg_sum, out_path):
    metrics = [
        ('dp_sex_mean', 'dp_sex_std', 'DP_sex (Demographic Parity)'),
        ('eo_sex_mean', 'eo_sex_std', 'EO_sex (Equal Opportunity)'),
        ('dp_age_mean', 'dp_age_std', 'DP_age'),
        ('eo_age_mean', 'eo_age_std', 'EO_age'),
        ('accuracy_mean', 'accuracy_std', 'Accuracy'),
        ('auc_mean',      'auc_std',      'AUC'),
    ]
    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    colors = {
        ('German Credit', 'Baseline'):   '#4C72B0',
        ('German Credit', 'Reweighing'): '#DD8452',
        ('South German',  'Baseline'):   '#55A868',
        ('South German',  'Reweighing'): '#C44E52',
    }
    labels = {
        ('German Credit', 'Baseline'):   'GC Baseline',
        ('German Credit', 'Reweighing'): 'GC +Reweight',
        ('South German',  'Baseline'):   'SG Baseline',
        ('South German',  'Reweighing'): 'SG +Reweight',
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        'Step 6: Cross-Dataset Generalization\nGerman Credit vs South German Credit (Groemping 2019)',
        fontsize=13, fontweight='bold'
    )

    x = np.arange(len(models))
    bar_w = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for ax, (mc, sc, title) in zip(axes.flat, metrics):
        for i, ((ds, cond), color) in enumerate(colors.items()):
            src = german_sum if ds == 'German Credit' else sg_sum
            sub = src[src['condition'] == cond].set_index('model')
            means = [sub.loc[m, mc] if m in sub.index else 0 for m in models]
            stds  = [sub.loc[m, sc] if m in sub.index else 0 for m in models]
            ax.bar(x + offsets[i] * bar_w, means, bar_w, yerr=stds, capsize=3,
                   color=color, alpha=0.85, label=labels[(ds, cond)],
                   error_kw={'elinewidth': 1.0})

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in colors.values()]
    fig.legend(handles, labels.values(), loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 6: Generalization — South German Credit")
    print("Groemping (2019) corrected dataset, N=1000")
    print("=" * 60)

    # ── Download & preprocess ─────────────────────────────────────────────
    print("\n[1] Loading South German Credit ...")
    raw_path = download_south_german()
    df = preprocess_south_german(raw_path)

    out_csv = 'data/south_german_credit_processed.csv'
    df.to_csv(out_csv, index=False)
    print(f"    Saved preprocessed: {out_csv}")

    X, y, sex, age = get_feature_matrix(df)

    print(f"    Samples: {len(y)}  |  Features: {X.shape[1]}")
    print(f"    Target   — Good: {(y==1).sum()}, Bad: {(y==0).sum()}")
    print(f"    sex_binary — Female(0): {(sex==0).sum()}, Male(1): {(sex==1).sum()}")
    print(f"    age_binary — Young(0): {(age==0).sum()}, Old(1): {(age==1).sum()}")
    print(f"    NOTE: famges=2 ambiguous coding acknowledged as limitation")

    # ── Run CV ────────────────────────────────────────────────────────────
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
            df_fold = run_cv(model_name, model, X, y, sex, age, condition)
            all_results.append(df_fold)

    results_df = pd.concat(all_results, ignore_index=True)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/step6_south_german_results.csv', index=False)
    print("\n    Saved: results/step6_south_german_results.csv")

    sg_summary = summarise(results_df)
    sg_summary.to_csv('results/step6_south_german_summary.csv', index=False)
    print("    Saved: results/step6_south_german_summary.csv")

    # ── Cross-dataset comparison ───────────────────────────────────────────
    print("\n[3] Cross-dataset comparison ...")
    german_sum = load_german_credit_summary()

    if german_sum is not None:
        table = build_cross_dataset_table(german_sum, sg_summary)
        table.to_csv('results/step6_cross_dataset_comparison.csv', index=False)
        print("    Saved: results/step6_cross_dataset_comparison.csv")
        print_cross_dataset_table(table)
        print("\n[4] Creating cross-dataset figure ...")
        make_cross_dataset_figure(german_sum, sg_summary,
                                  'figs/fig6_cross_dataset_comparison.png')
    else:
        print("    Skipping cross-dataset plot (Step 5 results not found).")
        print("    Run step5_reweighing_mitigation.py first, then re-run this script.")

    # ── Generalization verdict ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERALIZATION FINDINGS")
    print("=" * 60)

    for model_name in models:
        sg_base = sg_summary[(sg_summary['model'] == model_name) &
                             (sg_summary['condition'] == 'Baseline')]
        sg_rw   = sg_summary[(sg_summary['model'] == model_name) &
                             (sg_summary['condition'] == 'Reweighing')]
        if sg_base.empty or sg_rw.empty:
            continue
        sg_base, sg_rw = sg_base.iloc[0], sg_rw.iloc[0]
        delta_dp_sex = sg_rw['dp_sex_mean'] - sg_base['dp_sex_mean']
        delta_dp_age = sg_rw['dp_age_mean'] - sg_base['dp_age_mean']
        direction = "IMPROVED" if delta_dp_sex < 0 else "WORSENED"
        print(f"  {model_name}: DP_sex {direction} by {delta_dp_sex:+.4f}  |  "
              f"DP_age {'improved' if delta_dp_age < 0 else 'worsened'} "
              f"by {delta_dp_age:+.4f}")

    print("""
Thesis framing:
  Compare the above delta pattern with Step 5 (German Credit) to state
  whether the cross-dataset results are consistent. If RF improves and
  XGBoost is mixed on both datasets, the finding generalises.
  If patterns diverge, note dataset-specific factors (corrected coding,
  different famges=2 interpretation) as possible explanations.
""")
    print("=" * 60)
    print("Step 6 complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
