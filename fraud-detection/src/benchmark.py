"""
Low-Latency Inference Benchmarking System
===========================================
Measures single-transaction inference latency for each model.
Targets sub-2ms predictions to mirror Credit company's real-time requirements.
Runs 10,000 iterations and reports p50/p95/p99 percentile latencies.
"""

import numpy as np
import time
import json
import xgboost as xgb
import lightgbm as lgb
import torch
from pathlib import Path
import joblib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train_lstm import FraudLSTM


def benchmark_model(predict_fn, model_name, n_iterations=10000):
    """
    Benchmark single-prediction inference latency.

    Args:
        predict_fn: Callable that takes no args and performs one prediction
        model_name: Name for reporting
        n_iterations: Number of predictions to benchmark

    Returns:
        dict with latency statistics
    """
    # Warmup
    for _ in range(100):
        predict_fn()

    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        predict_fn()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to milliseconds

    latencies = np.array(latencies)

    stats = {
        'model': model_name,
        'n_iterations': n_iterations,
        'mean_ms': float(np.mean(latencies)),
        'median_ms': float(np.median(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'std_ms': float(np.std(latencies)),
        'under_2ms_pct': float((latencies < 2.0).mean() * 100),
    }

    return stats, latencies


def run_benchmarks():
    """Run latency benchmarks for all models."""
    print("=" * 70)
    print("  LOW-LATENCY INFERENCE BENCHMARK")
    print("  Target: < 2ms per transaction decision")
    print("=" * 70)

    models_dir = Path(__file__).parent.parent / 'models'
    results = []

    # Generate a sample input
    scaler = joblib.load(models_dir / 'scaler.pkl')
    n_features = scaler.n_features_in_
    sample = np.random.randn(1, n_features).astype(np.float32)

    # --- XGBoost Benchmark ---
    print("\n[1/3] Benchmarking XGBoost...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(models_dir / 'xgboost_fraud.json'))
    xgb_dmatrix = xgb.DMatrix(sample)

    def xgb_predict():
        return xgb_model.get_booster().predict(xgb_dmatrix)

    xgb_stats, xgb_lat = benchmark_model(xgb_predict, 'XGBoost')
    results.append(xgb_stats)

    # --- LightGBM Benchmark ---
    print("[2/3] Benchmarking LightGBM...")
    lgb_model = lgb.Booster(model_file=str(models_dir / 'lightgbm_fraud.txt'))

    def lgb_predict():
        return lgb_model.predict(sample)

    lgb_stats, lgb_lat = benchmark_model(lgb_predict, 'LightGBM')
    results.append(lgb_stats)

    # --- LSTM Benchmark ---
    print("[3/3] Benchmarking LSTM...")
    device = torch.device('cpu')  # CPU inference for fair comparison
    checkpoint = torch.load(models_dir / 'lstm_fraud.pt', map_location=device, weights_only=True)
    lstm_model = FraudLSTM(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    ).to(device)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()

    seq_sample = torch.randn(1, 10, n_features).float()

    def lstm_predict():
        with torch.no_grad():
            return lstm_model(seq_sample).item()

    lstm_stats, lstm_lat = benchmark_model(lstm_predict, 'LSTM')
    results.append(lstm_stats)

    # --- Print Results ---
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<12} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'<2ms':>8}")
    print("-" * 56)
    for r in results:
        status = "OK" if r['p95_ms'] < 2.0 else "!!"
        print(f"{r['model']:<12} {r['mean_ms']:>7.3f}ms {r['p50_ms']:>7.3f}ms "
              f"{r['p95_ms']:>7.3f}ms {r['p99_ms']:>7.3f}ms {r['under_2ms_pct']:>6.1f}% {status}")

    print(f"\n{'='*70}")
    print(f"  Target: All predictions under 2ms")
    for r in results:
        meets = r['p95_ms'] < 2.0
        emoji = "[PASS]" if meets else "[NOTE]"
        print(f"  {emoji} {r['model']}: p95={r['p95_ms']:.3f}ms "
              f"({r['under_2ms_pct']:.1f}% under 2ms)")

    # Save results
    results_path = models_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    run_benchmarks()
