"""
LightGBM Fraud Detection Model
================================
Trains a LightGBM classifier optimized for speed and fraud detection.
LightGBM's histogram-based approach provides faster training and inference.
"""

import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import FraudPreprocessor


def train_lightgbm():
    """Train LightGBM fraud detection model."""
    print("=" * 60)
    print("LIGHTGBM FRAUD DETECTION MODEL")
    print("=" * 60)

    # Prepare data
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_tree_data()

    # Configure LightGBM
    params = {
        'n_estimators': 200,
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'is_unbalance': True,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    print("\nTraining LightGBM model...")
    start_time = time.time()

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Save model
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'lightgbm_fraud.txt'
    model.booster_.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Feature importance (top 15)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:15]
    print("\n--- Top 15 Feature Importances ---")
    for i in sorted_idx:
        print(f"  {feature_names[i]:30s} {importance[i]}")

    return model, auc_roc, train_time


if __name__ == '__main__':
    train_lightgbm()
