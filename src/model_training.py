"""
Model Training Pipeline for XAI Rainfall Prediction
=====================================================
Trains, evaluates, compares, and persists:
  • Random Forest  (interpretable baseline)
  • XGBoost        (primary structured-data model)
  • MLP            (neural-network comparison)

Produces:
  • Saved models          → models/*.joblib
  • Metrics JSON          → models/metrics.json
  • Confusion matrices    → plots/confusion_matrix_*.png
  • ROC curve comparison  → plots/roc_curves_comparison.png
  • Bar-chart comparison  → plots/model_comparison.png
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report, roc_curve,
)
import matplotlib
matplotlib.use('Agg')          # headless backend
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  XGBoost not installed — skipping that model.")


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            random_state=config.RANDOM_STATE,
        ),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=config.RANDOM_STATE,
        )
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1_score':  round(f1_score(y_test, y_pred, zero_division=0), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metrics['roc_auc'] = round(roc_auc_score(y_test, y_prob), 4)
        metrics['fpr'] = fpr.tolist()
        metrics['tpr'] = tpr.tolist()

    # Pretty-print
    print(f"\n{'━' * 55}")
    print(f"  {name}")
    print(f"{'━' * 55}")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1-Score  : {metrics['f1_score']}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(classification_report(y_test, y_pred, target_names=['Non-Heavy', 'Heavy']))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Heavy', 'Heavy'],
                yticklabels=['Non-Heavy', 'Heavy'], ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR,
                        f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(all_metrics: dict) -> str:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, m in all_metrics.items():
        if 'fpr' in m:
            ax.plot(m['fpr'], m['tpr'], linewidth=2,
                    label=f"{name} (AUC = {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Model Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, 'roc_curves_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(all_metrics: dict) -> str:
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    names = list(all_metrics.keys())
    x = np.arange(len(metric_keys))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, name in enumerate(names):
        vals = [all_metrics[name].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.set_ylim(0, 1.12)
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, 'model_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_all():
    """Train every model, evaluate, save the best one."""
    print("\n" + "=" * 60)
    print("  MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load preprocessed splits
    X_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.csv'))
    X_val   = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'X_val.csv'))
    X_test  = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.csv')).values.ravel()
    y_val   = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'y_val.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.csv')).values.ravel()

    models = get_models()
    all_metrics = {}
    best_f1, best_name = -1.0, None
    validation_metrics = {}

    for name, model in models.items():
        print(f"\n🔧 Training {name} …")
        model.fit(X_train, y_train)

        # Use validation data for model selection. Keep the test set untouched
        # until every model has been trained and the winner has been selected.
        val_metrics = evaluate_model(model, X_val, y_val, f'{name} (validation)')
        validation_metrics[name] = val_metrics

        # Persist model
        slug = name.lower().replace(' ', '_')
        model_path = os.path.join(config.MODELS_DIR, f'{slug}.joblib')
        joblib.dump(model, model_path)
        print(f"  → saved {model_path}")

        if val_metrics['f1_score'] > best_f1:
            best_f1, best_name = val_metrics['f1_score'], name

    # Final reporting on the held-out test set. These scores do not determine
    # which model is selected.
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics[name] = metrics
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), name)

    # Best model copy
    if best_name:
        best = models[best_name]
        joblib.dump(best, os.path.join(config.MODELS_DIR, 'best_model.joblib'))
        print(f"\n★  Best model: {best_name}  (F1 = {best_f1:.4f})")

    # Comparison plots
    plot_roc_curves(all_metrics)
    plot_model_comparison(all_metrics)

    # Persist metrics (drop non-serialisable arrays for JSON)
    save_metrics = {}
    for name, m in all_metrics.items():
        save_metrics[name] = {k: v for k, v in m.items() if k not in ('fpr', 'tpr')}
    save_metrics['best_model'] = best_name
    save_metrics['selection_metric'] = 'validation_f1_score'
    save_metrics['validation'] = {
        name: {key: value for key, value in metrics.items() if key not in ('fpr', 'tpr')}
        for name, metrics in validation_metrics.items()
    }

    metrics_path = os.path.join(config.MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)

    print(f"\n✔ Metrics  → {metrics_path}")
    print(f"✔ Plots    → {config.PLOTS_DIR}/")
    return models, all_metrics, best_name


if __name__ == '__main__':
    train_all()
