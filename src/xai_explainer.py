"""
XAI Explainability Module
==========================
Uses SHAP (SHapley Additive exPlanations) to provide:
  • Global feature importance rankings
  • SHAP summary (beeswarm) plots
  • Instance-level force-plot explanations
  • Feature dependence plots
  • Model failure analysis — when and why the model gets it wrong
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_artefacts():
    """Load the best model, scaler, and test data."""
    model  = joblib.load(os.path.join(config.MODELS_DIR, 'best_model.joblib'))
    scaler = joblib.load(os.path.join(config.MODELS_DIR, 'scaler.joblib'))
    X_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.csv')).values.ravel()
    return model, scaler, X_test, y_test


def get_explainer(model, X_background):
    """Create the right SHAP explainer for the model type."""
    model_type = type(model).__name__

    if model_type in ('RandomForestClassifier', 'GradientBoostingClassifier'):
        return shap.TreeExplainer(model)
    elif model_type in ('XGBClassifier',):
        return shap.TreeExplainer(model)
    else:
        # Kernel explainer for anything else (MLP, etc.)
        # Use a small background sample for speed
        bg = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(model.predict_proba, bg)


# ─────────────────────────────────────────────────────────────────────────────
# Global explanations
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X_data, max_samples: int = 500):
    """Compute SHAP values; returns (explainer, shap_values)."""
    if len(X_data) > max_samples:
        X_sample = X_data.sample(max_samples, random_state=config.RANDOM_STATE)
    else:
        X_sample = X_data

    explainer = get_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers, tree explainer returns [class0, class1] or 3D array (samples, features, 2)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]   # positive-class explanations
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[-1] == 2:
        shap_values = shap_values[:, :, 1]

    return explainer, shap_values, X_sample


def plot_feature_importance(shap_values, X_sample) -> str:
    """Global mean |SHAP| bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    col_names = list(X_sample.columns)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    features = [col_names[int(i)] for i in order]
    labels = [config.FEATURE_LABELS.get(f, f) for f in features]
    vals = [float(mean_abs[int(i)]) for i in order]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(range(len(features)), vals[::-1], color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Global Feature Importance (SHAP)', fontweight='bold')
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, 'shap_feature_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")
    return path


def plot_summary(shap_values, X_sample) -> str:
    """Beeswarm / dot summary plot."""
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
    plt.title('SHAP Summary Plot', fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, 'shap_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  → {path}")
    return path


def plot_dependence(shap_values, X_sample, feature: str) -> str:
    """Dependence plot for a single feature."""
    fig = plt.figure(figsize=(9, 6))
    shap.dependence_plot(feature, shap_values, X_sample, show=False)
    plt.title(f'SHAP Dependence — {config.FEATURE_LABELS.get(feature, feature)}',
              fontweight='bold', pad=10)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f'shap_dependence_{feature}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Instance-level explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_instance(model, scaler, raw_features: dict) -> dict:
    """
    Given raw (unscaled) feature values, return prediction + SHAP breakdown.

    Returns
    -------
    dict with keys: prediction, confidence, shap_values, feature_contributions
    """
    df = pd.DataFrame([raw_features])[config.FEATURES]
    X_scaled = pd.DataFrame(scaler.transform(df), columns=config.FEATURES)

    # Prediction
    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    confidence = float(max(proba))

    # SHAP
    explainer = get_explainer(model, X_scaled)
    sv = explainer.shap_values(X_scaled)
    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3 and sv.shape[-1] == 2:
        sv = sv[:, :, 1]
    if isinstance(sv, np.ndarray) and sv.ndim == 2:
        sv = sv[0]
    elif isinstance(sv, list) and len(sv) > 0 and hasattr(sv[0], '__len__'):
        sv = sv[0]

    contributions = []
    for feat, val in zip(config.FEATURES, sv):
        contributions.append({
            'feature': feat,
            'label': config.FEATURE_LABELS.get(feat, feat),
            'shap_value': round(float(val), 5),
            'raw_value': round(float(raw_features.get(feat, 0)), 2),
        })
    contributions.sort(key=lambda c: abs(c['shap_value']), reverse=True)

    return {
        'prediction': pred,
        'prediction_label': 'Heavy Rain' if pred == 1 else 'Non-Heavy Rain',
        'confidence': round(confidence, 4),
        'shap_values': sv.tolist(),
        'feature_contributions': contributions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model failure analysis
# ─────────────────────────────────────────────────────────────────────────────

def failure_analysis(model, X_test, y_test, shap_values, X_sample) -> dict:
    """
    Identify misclassified instances and analyse common patterns.

    Returns a dict with:
        total_errors, false_positives, false_negatives,
        failure_patterns (feature-range conditions where model struggles)
    """
    y_pred = model.predict(X_test)
    mask_wrong = y_pred != y_test

    fp_mask = (y_pred == 1) & (y_test == 0)
    fn_mask = (y_pred == 0) & (y_test == 1)

    result = {
        'total_test':      int(len(y_test)),
        'total_errors':    int(mask_wrong.sum()),
        'error_rate':      round(float(mask_wrong.mean()), 4),
        'false_positives': int(fp_mask.sum()),
        'false_negatives': int(fn_mask.sum()),
        'failure_patterns': [],
    }

    # Analyse feature distributions in errors vs correct
    X_wrong = X_test[mask_wrong]
    X_right = X_test[~mask_wrong]

    patterns = []
    for feat in config.FEATURES:
        if feat in X_test.columns:
            wrong_mean = float(X_wrong[feat].mean()) if len(X_wrong) > 0 else 0
            right_mean = float(X_right[feat].mean()) if len(X_right) > 0 else 0
            wrong_std  = float(X_wrong[feat].std())  if len(X_wrong) > 1 else 0
            diff = abs(wrong_mean - right_mean)

            patterns.append({
                'feature': feat,
                'label': config.FEATURE_LABELS.get(feat, feat),
                'wrong_mean': round(wrong_mean, 3),
                'correct_mean': round(right_mean, 3),
                'difference': round(diff, 3),
                'wrong_std': round(wrong_std, 3),
            })

    # Sort by largest difference
    patterns.sort(key=lambda p: p['difference'], reverse=True)
    result['failure_patterns'] = patterns[:5]   # top-5 discriminating features

    # False-negative analysis (missed heavy rain — the dangerous kind)
    fn_data = X_test[fn_mask]
    if len(fn_data) > 0:
        result['missed_heavy_rain'] = {
            'count': int(fn_mask.sum()),
            'description': 'Instances where heavy rain occurred but the model predicted non-heavy.',
            'avg_features': {
                feat: round(float(fn_data[feat].mean()), 2)
                for feat in config.FEATURES if feat in fn_data.columns
            },
        }
    else:
        result['missed_heavy_rain'] = {'count': 0, 'description': 'No missed heavy rain events.'}

    # Save failure report
    report_path = os.path.join(config.MODELS_DIR, 'failure_analysis.json')
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  → Failure report: {report_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plot failure analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_failure_patterns(failure_report: dict) -> str:
    """Bar chart of feature differences between correct and incorrect predictions."""
    patterns = failure_report.get('failure_patterns', [])
    if not patterns:
        return ''

    fig, ax = plt.subplots(figsize=(10, 6))
    features = [p['label'] for p in patterns]
    diffs = [p['difference'] for p in patterns]

    colors = ['#e74c3c' if d > 0.1 else '#f39c12' if d > 0.05 else '#27ae60' for d in diffs]
    ax.barh(features[::-1], diffs[::-1], color=colors[::-1])
    ax.set_xlabel('|Mean Difference| (Wrong vs Correct)')
    ax.set_title('Top Features Where Model Fails', fontweight='bold')
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, 'failure_patterns.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_xai_pipeline():
    """Full XAI analysis pipeline."""
    print("\n" + "=" * 60)
    print("  XAI EXPLAINABILITY PIPELINE")
    print("=" * 60)

    model, scaler, X_test, y_test = _load_artefacts()

    print("\n📊 Computing SHAP values …")
    explainer, shap_vals, X_sample = compute_shap_values(model, X_test)

    print("\n📈 Generating global explanation plots …")
    plot_feature_importance(shap_vals, X_sample)
    plot_summary(shap_vals, X_sample)

    # Dependence plots for top-3 features
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top3 = np.argsort(mean_abs)[::-1][:3]
    for idx in top3:
        feat = X_sample.columns[idx]
        plot_dependence(shap_vals, X_sample, feat)

    print("\n🔍 Running failure analysis …")
    report = failure_analysis(model, X_test, y_test, shap_vals, X_sample)
    plot_failure_patterns(report)

    print(f"\n✔ XAI pipeline complete.")
    print(f"  Total errors: {report['total_errors']} / {report['total_test']}")
    print(f"  False Negatives (missed heavy rain): {report['false_negatives']}")
    print(f"  False Positives: {report['false_positives']}")

    return explainer, shap_vals, X_sample, report


if __name__ == '__main__':
    run_xai_pipeline()
