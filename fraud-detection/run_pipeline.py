"""
End-to-End Fraud Detection Pipeline Runner
=============================================
Executes the complete pipeline in sequence:
1. Generate synthetic dataset
2. Train XGBoost model
3. Train LightGBM model
4. Train LSTM model
5. Evaluate all models
6. Benchmark inference latency
"""

import time
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_pipeline():
    """Execute the complete fraud detection pipeline."""
    print("=" * 70)
    print("  HIGH-VELOCITY FRAUD DETECTION PIPELINE")
    print("  Credit company-Inspired Real-Time Fraud Detection System")
    print("=" * 70)

    total_start = time.time()

    # Step 1: Generate Dataset
    print("\n\n" + "-" * 70)
    print("  STEP 1/6: GENERATING SYNTHETIC TRANSACTION DATASET")
    print("-" * 70)
    from data.generate_dataset import main as generate_data
    generate_data()

    # Step 2: Train XGBoost
    print("\n\n" + "-" * 70)
    print("  STEP 2/6: TRAINING XGBOOST MODEL")
    print("-" * 70)
    from src.train_xgboost import train_xgboost
    xgb_model, xgb_auc, xgb_time = train_xgboost()

    # Step 3: Train LightGBM
    print("\n\n" + "-" * 70)
    print("  STEP 3/6: TRAINING LIGHTGBM MODEL")
    print("-" * 70)
    from src.train_lightgbm import train_lightgbm
    lgb_model, lgb_auc, lgb_time = train_lightgbm()

    # Step 4: Train LSTM
    print("\n\n" + "-" * 70)
    print("  STEP 4/6: TRAINING LSTM MODEL")
    print("-" * 70)
    from src.train_lstm import train_lstm
    lstm_model, lstm_auc, lstm_time = train_lstm()

    # Step 5: Evaluate All Models
    print("\n\n" + "-" * 70)
    print("  STEP 5/6: EVALUATING ALL MODELS")
    print("-" * 70)
    from src.evaluate import evaluate_all_models
    evaluate_all_models()

    # Step 6: Benchmark Latency
    print("\n\n" + "-" * 70)
    print("  STEP 6/6: BENCHMARKING INFERENCE LATENCY")
    print("-" * 70)
    from src.benchmark import run_benchmarks
    benchmark_results = run_benchmarks()

    # Final Summary
    total_time = time.time() - total_start
    print("\n\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total pipeline time: {total_time:.1f}s")
    print()
    print("  Training Times:")
    print(f"    XGBoost:  {xgb_time:.1f}s  |  AUC-ROC: {xgb_auc:.4f}")
    print(f"    LightGBM: {lgb_time:.1f}s  |  AUC-ROC: {lgb_auc:.4f}")
    print(f"    LSTM:     {lstm_time:.1f}s  |  AUC-ROC: {lstm_auc:.4f}")
    print()
    print("  Latency Benchmarks (p95):")
    for r in benchmark_results:
        status = "PASS" if r['p95_ms'] < 2.0 else "NOTE"
        print(f"    {r['model']:<10s}: {r['p95_ms']:.3f}ms  [{status}]")
    print()
    print("  Outputs:")
    print("    Models:  models/")
    print("    Results: results/")
    print("=" * 70)

    print("\n  To launch the dashboard: python app/server.py")
    print("  Then open: http://localhost:5000\n")


if __name__ == '__main__':
    run_pipeline()
