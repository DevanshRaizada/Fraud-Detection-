"""
Flask API Server for Real-Time Fraud Detection
================================================
Serves the web dashboard and provides API endpoints for
real-time transaction fraud prediction using all 3 models.
"""

import numpy as np
import json
import time
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import xgboost as xgb
import lightgbm as lgb
import torch
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train_lstm import FraudLSTM

app = Flask(__name__)
CORS(app)

# Global model references
models = {}
scaler = None
feature_columns = None


def load_models():
    """Load all trained models into memory."""
    global models, scaler, feature_columns

    models_dir = Path(__file__).parent.parent / 'models'

    print("Loading models...")

    # Load scaler and feature info
    scaler = joblib.load(models_dir / 'scaler.pkl')
    feature_columns = joblib.load(models_dir / 'feature_columns.pkl')

    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(models_dir / 'xgboost_fraud.json'))
    models['xgboost'] = xgb_model
    print("  [OK] XGBoost loaded")

    # LightGBM
    lgb_model = lgb.Booster(model_file=str(models_dir / 'lightgbm_fraud.txt'))
    models['lightgbm'] = lgb_model
    print("  [OK] LightGBM loaded")

    # LSTM
    device = torch.device('cpu')
    checkpoint = torch.load(models_dir / 'lstm_fraud.pt', map_location=device, weights_only=True)
    lstm_model = FraudLSTM(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    models['lstm'] = lstm_model
    print("  [OK] LSTM loaded")

    print("All models loaded successfully!\n")


def generate_random_transaction():
    """Generate a random transaction for simulation."""
    rng = np.random.default_rng()
    n_features = len(feature_columns)
    features = rng.standard_normal(n_features).astype(np.float32)

    # Make it more realistic
    is_fraud = rng.random() < 0.15  # Higher fraud rate for demo purposes

    if is_fraud:
        features[0] -= rng.uniform(2, 4)   # V1
        features[2] -= rng.uniform(1.5, 3)  # V3
        features[13] -= rng.uniform(1.5, 3) # V14

    amount = float(np.exp(rng.uniform(1, 8))) if is_fraud else float(np.exp(rng.uniform(1, 5)))
    hour = int(rng.choice(range(24)))

    categories = ['grocery', 'gas_station', 'restaurant', 'online', 'retail', 'travel', 'entertainment']
    category = rng.choice(categories)

    return {
        'features': features,
        'amount': round(amount, 2),
        'hour': hour,
        'category': category,
        'card_last4': f"{rng.integers(1000, 9999)}",
        'merchant': f"Merchant_{rng.integers(1000, 9999)}",
    }


@app.route('/')
def dashboard():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run fraud prediction on a transaction using all models."""
    data = request.json or {}

    if 'features' in data:
        features = np.array(data['features']).reshape(1, -1).astype(np.float32)
    else:
        txn = generate_random_transaction()
        features = txn['features'].reshape(1, -1)
        data = txn

    results = {}

    # XGBoost prediction
    start = time.perf_counter()
    xgb_dmatrix = xgb.DMatrix(features)
    xgb_prob = float(models['xgboost'].get_booster().predict(xgb_dmatrix)[0])
    xgb_time = (time.perf_counter() - start) * 1000
    results['xgboost'] = {'probability': xgb_prob, 'latency_ms': round(xgb_time, 3)}

    # LightGBM prediction
    start = time.perf_counter()
    lgb_prob = float(models['lightgbm'].predict(features)[0])
    lgb_time = (time.perf_counter() - start) * 1000
    results['lightgbm'] = {'probability': lgb_prob, 'latency_ms': round(lgb_time, 3)}

    # LSTM prediction
    start = time.perf_counter()
    seq = torch.FloatTensor(np.tile(features, (1, 10, 1)))
    with torch.no_grad():
        lstm_prob = float(models['lstm'](seq).item())
    lstm_time = (time.perf_counter() - start) * 1000
    results['lstm'] = {'probability': lstm_prob, 'latency_ms': round(lstm_time, 3)}

    # Determine consensus
    avg_prob = np.mean([xgb_prob, lgb_prob, lstm_prob])
    is_fraud = avg_prob >= 0.5

    return jsonify({
        'transaction': {
            'amount': data.get('amount', round(float(np.exp(np.random.uniform(1, 6))), 2)),
            'hour': data.get('hour', int(np.random.randint(0, 24))),
            'category': data.get('category', 'unknown'),
            'card_last4': data.get('card_last4', '0000'),
            'merchant': data.get('merchant', 'Unknown'),
        },
        'models': results,
        'consensus': {
            'probability': round(float(avg_prob), 4),
            'is_fraud': bool(is_fraud),
            'decision': 'BLOCK' if is_fraud else 'APPROVE',
        }
    })


@app.route('/api/simulate', methods=['GET'])
def simulate():
    """Simulate a random transaction and predict."""
    txn = generate_random_transaction()
    features = txn['features'].reshape(1, -1)

    results = {}

    # XGBoost
    start = time.perf_counter()
    xgb_dmatrix = xgb.DMatrix(features)
    xgb_prob = float(models['xgboost'].get_booster().predict(xgb_dmatrix)[0])
    xgb_time = (time.perf_counter() - start) * 1000
    results['xgboost'] = {'probability': round(xgb_prob, 4), 'latency_ms': round(xgb_time, 3)}

    # LightGBM
    start = time.perf_counter()
    lgb_prob = float(models['lightgbm'].predict(features)[0])
    lgb_time = (time.perf_counter() - start) * 1000
    results['lightgbm'] = {'probability': round(lgb_prob, 4), 'latency_ms': round(lgb_time, 3)}

    # LSTM
    start = time.perf_counter()
    seq = torch.FloatTensor(np.tile(features, (1, 10, 1)))
    with torch.no_grad():
        lstm_prob = float(models['lstm'](seq).item())
    lstm_time = (time.perf_counter() - start) * 1000
    results['lstm'] = {'probability': round(lstm_prob, 4), 'latency_ms': round(lstm_time, 3)}

    avg_prob = np.mean([xgb_prob, lgb_prob, lstm_prob])
    is_fraud = avg_prob >= 0.5

    return jsonify({
        'transaction': {
            'amount': txn['amount'],
            'hour': txn['hour'],
            'category': txn['category'],
            'card_last4': txn['card_last4'],
            'merchant': txn['merchant'],
        },
        'models': results,
        'consensus': {
            'probability': round(float(avg_prob), 4),
            'is_fraud': bool(is_fraud),
            'decision': 'BLOCK' if is_fraud else 'APPROVE',
        }
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Return model evaluation stats and benchmark results."""
    results_dir = Path(__file__).parent.parent / 'results'
    models_dir = Path(__file__).parent.parent / 'models'

    data = {}

    # Load evaluation results
    eval_path = results_dir / 'evaluation_results.json'
    if eval_path.exists():
        with open(eval_path) as f:
            data['evaluation'] = json.load(f)

    # Load benchmark results
    bench_path = models_dir / 'benchmark_results.json'
    if bench_path.exists():
        with open(bench_path) as f:
            data['benchmarks'] = json.load(f)

    return jsonify(data)


if __name__ == '__main__':
    load_models()
    print("Starting Fraud Detection Dashboard...")
    print("Open: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
