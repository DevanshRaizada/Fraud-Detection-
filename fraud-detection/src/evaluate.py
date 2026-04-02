"""
Model Evaluation & Comparison
===============================
Evaluates all trained models and generates comparison metrics,
confusion matrices, ROC curves, and PR curves.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import torch
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import FraudPreprocessor
from src.train_lstm import FraudLSTM


def evaluate_all_models():
    """Evaluate and compare all trained models."""
    print("=" * 60)
    print("MODEL EVALUATION & COMPARISON")
    print("=" * 60)

    models_dir = Path(__file__).parent.parent / 'models'
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # -- Prepare test data --
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_tree_data(apply_smote=False)

    results = {}

    # -- XGBoost --
    print("\n[1/3] Evaluating XGBoost...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(models_dir / 'xgboost_fraud.json'))
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)
    results['XGBoost'] = {
        'y_prob': y_prob_xgb,
        'y_pred': y_pred_xgb,
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'auc_roc': roc_auc_score(y_test, y_prob_xgb),
        'pr_auc': average_precision_score(y_test, y_prob_xgb),
    }

    # -- LightGBM --
    print("[2/3] Evaluating LightGBM...")
    lgb_model = lgb.Booster(model_file=str(models_dir / 'lightgbm_fraud.txt'))
    y_prob_lgb = lgb_model.predict(X_test)
    y_pred_lgb = (y_prob_lgb >= 0.5).astype(int)
    results['LightGBM'] = {
        'y_prob': y_prob_lgb,
        'y_pred': y_pred_lgb,
        'precision': precision_score(y_test, y_pred_lgb),
        'recall': recall_score(y_test, y_pred_lgb),
        'f1': f1_score(y_test, y_pred_lgb),
        'auc_roc': roc_auc_score(y_test, y_prob_lgb),
        'pr_auc': average_precision_score(y_test, y_prob_lgb),
    }

    # -- LSTM --
    print("[3/3] Evaluating LSTM...")
    _, X_test_seq, _, y_test_seq = preprocessor.prepare_lstm_data(sequence_length=10)
    device = torch.device('cpu')
    checkpoint = torch.load(models_dir / 'lstm_fraud.pt', map_location=device, weights_only=True)
    lstm_model = FraudLSTM(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_prob_lstm = lstm_model(X_test_tensor).numpy()
    y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)
    results['LSTM'] = {
        'y_prob': y_prob_lstm,
        'y_pred': y_pred_lstm,
        'precision': precision_score(y_test_seq, y_pred_lstm),
        'recall': recall_score(y_test_seq, y_pred_lstm),
        'f1': f1_score(y_test_seq, y_pred_lstm),
        'auc_roc': roc_auc_score(y_test_seq, y_prob_lstm),
        'pr_auc': average_precision_score(y_test_seq, y_prob_lstm),
    }

    # -- Print Comparison Table --
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10} {'PR-AUC':>10}")
    print("-" * 62)
    for name, m in results.items():
        print(f"{name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['auc_roc']:>10.4f} {m['pr_auc']:>10.4f}")

    # -- Generate Plots --
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Fraud Detection Model Comparison', fontsize=16, fontweight='bold')
    colors = {'XGBoost': '#2196F3', 'LightGBM': '#4CAF50', 'LSTM': '#FF9800'}

    # 1. ROC Curves
    ax = axes[0, 0]
    for name, m in results.items():
        y_true = y_test if name != 'LSTM' else y_test_seq
        fpr, tpr, _ = roc_curve(y_true, m['y_prob'])
        ax.plot(fpr, tpr, color=colors[name], linewidth=2,
                label=f"{name} (AUC={m['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. PR Curves
    ax = axes[0, 1]
    for name, m in results.items():
        y_true = y_test if name != 'LSTM' else y_test_seq
        pre, rec, _ = precision_recall_curve(y_true, m['y_prob'])
        ax.plot(rec, pre, color=colors[name], linewidth=2,
                label=f"{name} (PR-AUC={m['pr_auc']:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Confusion Matrices
    ax = axes[1, 0]
    y_true_combined = y_test
    cm = confusion_matrix(y_true_combined, results['XGBoost']['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    ax.set_title('XGBoost Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # 4. Metric Comparison Bar Chart
    ax = axes[1, 1]
    metrics = ['precision', 'recall', 'f1', 'auc_roc']
    x = np.arange(len(metrics))
    width = 0.25
    for i, (name, m) in enumerate(results.items()):
        vals = [m[metric] for metric in metrics]
        ax.bar(x + i * width, vals, width, label=name, color=colors[name])
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Precision', 'Recall', 'F1', 'AUC-ROC'])
    ax.set_title('Model Metric Comparison')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plots saved to: {plot_path}")

    # Save metrics JSON
    metrics_json = {name: {k: float(v) for k, v in m.items() if k not in ['y_prob', 'y_pred']}
                    for name, m in results.items()}
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to: {output_dir / 'evaluation_results.json'}")

    return results


if __name__ == '__main__':
    evaluate_all_models()
