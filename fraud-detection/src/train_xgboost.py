"""
XGBoost Fraud Detection Model
===============================
Trains an XGBoost classifier optimized for fraud detection with
class imbalance handling via scale_pos_weight.
"""

import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import FraudPreprocessor


def train_xgboost():
    """Train XGBoost fraud detection model."""
    print("=" * 60)
    print("XGBOOST FRAUD DETECTION MODEL")
    print("=" * 60)

    # Prepare data
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_tree_data()

    # Calculate class weight ratio
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    scale_pos_weight = n_legit / n_fraud

    print(f"\nScale pos weight: {scale_pos_weight:.2f}")

    # Configure XGBoost
    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',         # Fast histogram-based method
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    print("\nTraining XGBoost model...")
    start_time = time.time()

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
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
    model_path = models_dir / 'xgboost_fraud.json'
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Feature importance (top 15)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:15]
    print("\n--- Top 15 Feature Importances ---")
    for i in sorted_idx:
        print(f"  {feature_names[i]:30s} {importance[i]:.4f}")

    return model, auc_roc, train_time


if __name__ == '__main__':
    train_xgboost()
