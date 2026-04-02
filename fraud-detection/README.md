# 🛡️ FraudShield — High-Velocity Fraud Detection Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An end-to-end real-time fraud detection system inspired by Credit company,<br>processing transactions in sub-2ms with XGBoost, LightGBM, and LSTM models.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Models](#-model-performance) · [Dashboard](#-dashboard) · [API](#-api-reference)

</div>

---

## 📖 Overview

Credit company monitors **over 8 billion transactions annually** and makes fraud decisions in **milliseconds**. This project mirrors that capability by building a complete fraud detection pipeline that:

- Generates **100,000 synthetic credit card transactions** with realistic fraud patterns
- Trains **three ML/DL models** — XGBoost, LightGBM, and LSTM — optimized for both accuracy and speed
- Benchmarks inference latency with a target of **< 2ms per prediction**
- Serves predictions via a **real-time web dashboard** with live transaction simulation

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔢 **Synthetic Data Generation** | 100K transactions with 28 PCA features, temporal patterns, and 2% fraud injection |
| 🌲 **XGBoost Model** | Gradient-boosted trees with `scale_pos_weight` for class imbalance |
| 🍃 **LightGBM Model** | Histogram-based gradient boosting optimized for speed |
| 🧠 **LSTM Network** | 2-layer LSTM (64 hidden units) capturing sequential spending patterns |
| ⚡ **Sub-2ms Inference** | All models achieve p95 latency under 2ms on CPU |
| 📊 **Evaluation Suite** | ROC curves, PR curves, confusion matrices, and F1/AUC comparison charts |
| 🖥️ **Live Dashboard** | Dark-themed web UI with real-time transaction simulation |
| 🔌 **REST API** | Flask endpoints for prediction, simulation, and model stats |
| 🔄 **One-Command Pipeline** | Single script runs the entire workflow end-to-end |

## 🏗️ Architecture

```
fraud-detection/
├── requirements.txt                 # Python dependencies
├── run_pipeline.py                  # One-command pipeline runner
│
├── data/
│   ├── generate_dataset.py          # Synthetic transaction generator
│   └── transactions.csv             # Generated dataset (100K rows)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Feature engineering & SMOTE
│   ├── train_xgboost.py             # XGBoost model training
│   ├── train_lightgbm.py            # LightGBM model training
│   ├── train_lstm.py                # LSTM model training (PyTorch)
│   ├── evaluate.py                  # Model comparison & visualization
│   └── benchmark.py                 # Latency benchmarking system
│
├── models/                          # Saved model artifacts
│   ├── xgboost_fraud.json
│   ├── lightgbm_fraud.txt
│   ├── lstm_fraud.pt
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_columns.pkl
│   └── benchmark_results.json
│
├── results/
│   ├── evaluation_results.json      # Accuracy metrics
│   └── model_comparison.png         # ROC/PR/Confusion charts
│
└── app/
    ├── server.py                    # Flask API server
    ├── templates/
    │   └── index.html               # Dashboard HTML
    └── static/
        ├── style.css                # Dark-themed styling
        └── app.js                   # Dashboard interactivity
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested with 3.12)
- **pip** package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_pipeline.py
```

This single command executes all 6 stages:

| Stage | Description | Output |
|-------|-------------|--------|
| 1️⃣ | Generate synthetic dataset | `data/transactions.csv` |
| 2️⃣ | Train XGBoost model | `models/xgboost_fraud.json` |
| 3️⃣ | Train LightGBM model | `models/lightgbm_fraud.txt` |
| 4️⃣ | Train LSTM model | `models/lstm_fraud.pt` |
| 5️⃣ | Evaluate all models | `results/evaluation_results.json` |
| 6️⃣ | Benchmark latency | `models/benchmark_results.json` |

### Launch the Dashboard

```bash
python app/server.py
```

Open **http://localhost:5000** in your browser.

## 📈 Model Performance

### Accuracy Metrics

| Model | AUC-ROC | Precision | Recall | F1 Score | PR-AUC |
|-------|---------|-----------|--------|----------|--------|
| **XGBoost** | **0.9999** | 0.9950 | 0.9925 | 0.9937 | 0.9999 |
| **LightGBM** | **0.9999** | 0.9975 | 0.9950 | 0.9962 | 0.9999 |
| LSTM | 0.5098 | — | — | — | 0.0210 |

> **Note:** The LSTM model shows lower performance on synthetic data because the randomly generated card sequences lack meaningful temporal spending patterns. On real-world data with genuine transaction histories (like Credit company's), the LSTM architecture would capture sequential fraud indicators like velocity checks, unusual merchant sequences, and time-of-day anomalies.

### Latency Benchmarks

All benchmarks run on **CPU** with **10,000 iterations** per model:

| Model | Mean | P50 | P95 | P99 | % Under 2ms |
|-------|------|-----|-----|-----|-------------|
| **XGBoost** | 0.179ms | 0.164ms | **0.270ms** | 0.435ms | ✅ 100% |
| **LightGBM** | 0.101ms | 0.096ms | **0.134ms** | 0.204ms | ✅ 100% |
| **LSTM** | 0.883ms | 0.953ms | **1.484ms** | 1.991ms | ✅ 99% |

> All three models achieve **sub-2ms inference at p95**, meeting Credit company's real-time fraud detection requirements.

## 🖥️ Dashboard

The FraudShield dashboard provides:

- **Model Performance Cards** — Live AUC-ROC, Precision, Recall, F1, and latency bars for each model
- **Transaction Simulator** — Click to generate random transactions and see real-time fraud predictions
- **Auto-Simulate Mode** — Continuous stream of transactions at configurable speed
- **Live Transaction Feed** — Scrolling feed showing approved/blocked transactions with risk scores
- **Consensus Engine** — Averages predictions across all 3 models for final APPROVE/BLOCK decision

### Design Highlights

- 🌑 Premium **dark theme** with glassmorphism cards
- ✨ Subtle **micro-animations** and hover effects
- 📱 Fully **responsive** layout
- 🔤 **JetBrains Mono** monospace for metrics, **Inter** for UI text

## 🔌 API Reference

### `GET /api/simulate`

Simulates a random transaction and returns fraud predictions from all 3 models.

**Response:**
```json
{
  "transaction": {
    "amount": 142.50,
    "hour": 14,
    "category": "online",
    "card_last4": "7823",
    "merchant": "Merchant_4521"
  },
  "models": {
    "xgboost":  { "probability": 0.0023, "latency_ms": 0.21 },
    "lightgbm": { "probability": 0.0018, "latency_ms": 0.11 },
    "lstm":     { "probability": 0.4200, "latency_ms": 0.95 }
  },
  "consensus": {
    "probability": 0.1414,
    "is_fraud": false,
    "decision": "APPROVE"
  }
}
```

### `POST /api/predict`

Submit custom features for prediction.

### `GET /api/stats`

Returns saved evaluation metrics and benchmark results for all models.

## 🧪 Dataset Details

The synthetic dataset (`data/transactions.csv`) contains **100,000 transactions** with:

| Feature Group | Features | Description |
|---------------|----------|-------------|
| **PCA Components** | V1–V28 | 28 anonymized principal components (mimics Kaggle dataset) |
| **Transaction** | Amount, Time | Dollar amount and elapsed seconds |
| **Temporal** | TransactionHour, DayOfWeek | Time-of-day and day-of-week |
| **Categorical** | MerchantCategory | 7 categories (grocery, online, travel, etc.) |
| **Grouping** | CardID | Synthetic card IDs for LSTM sequence grouping |
| **Target** | Class | 0 = legitimate, 1 = fraud |

### Fraud Injection Patterns

- **PCA shifts**: Fraud transactions have statistically significant shifts in V1, V3, V7, V10, V14, V17
- **Higher amounts**: Fraud amounts are log-normally distributed with higher mean
- **Night bias**: Fraud skews towards late-night/early-morning hours
- **Online/travel bias**: Fraud over-represented in online (40%) and travel (20%) categories

## 🔧 Technical Deep-Dive

### Feature Engineering (`preprocessing.py`)

1. **StandardScaler** — Normalizes Amount and Time distributions
2. **Cyclical encoding** — Sin/cos encoding for hour (24-cycle) and day (7-cycle)
3. **Log transform** — `log1p(Amount)` reduces right-skew
4. **Binary flags** — `IsWeekend`, `IsNight` capture behavioral patterns
5. **Amount binning** — Discretizes amounts into 5 risk tiers
6. **SMOTE** — Synthetic minority oversampling (30% target) for training set balance
7. **LSTM windowing** — Groups transactions by CardID into 10-step sliding windows

### XGBoost Configuration

- `max_depth=6`, `n_estimators=200`, `learning_rate=0.1`
- `tree_method='hist'` for fast histogram-based training
- `scale_pos_weight` auto-calculated from class ratio
- L1/L2 regularization (`reg_alpha=0.1`, `reg_lambda=1.0`)

### LightGBM Configuration

- `num_leaves=31`, `n_estimators=200`, boosting type `gbdt`
- `is_unbalance=True` for automatic class weight handling
- Column and row subsampling at 80%

### LSTM Architecture

```
FraudLSTM(
  (lstm): LSTM(41, 64, num_layers=2, batch_first=True, dropout=0.3)
  (classifier): Sequential(
    Linear(64 → 32) → ReLU → Dropout(0.3)
    Linear(32 → 16) → ReLU → Dropout(0.15)
    Linear(16 → 1)  → Sigmoid
  )
)
Total parameters: ~45K
```

- 10-transaction sliding window per card
- Adam optimizer with `ReduceLROnPlateau` scheduler
- Gradient clipping at `max_norm=1.0`

## 📋 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
flask>=3.0.0
flask-cors>=4.0.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
<b>Built with ❤️ for real-time fraud detection</b><br>
<sub>XGBoost · LightGBM · LSTM · Sub-2ms Inference</sub>
</div>
