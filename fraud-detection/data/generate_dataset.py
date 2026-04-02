"""
Synthetic Credit Card Transaction Dataset Generator
====================================================
Generates ~100K anonymized credit card transactions with realistic fraud patterns.
Mimics the structure of the famous Kaggle credit card fraud dataset with 28 PCA
features (V1-V28) plus Amount, Time, and engineered temporal features.

Fraud rate: ~2% (class imbalance mirrors real-world scenarios)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path


def generate_pca_features(n_samples, is_fraud, rng):
    """Generate 28 PCA-transformed features with fraud-specific patterns."""
    features = rng.standard_normal((n_samples, 28))

    if is_fraud:
        # Fraud transactions have distinctive patterns in certain components
        features[:, 0] -= rng.uniform(2.0, 4.0, n_samples)    # V1 shifts negative
        features[:, 1] += rng.uniform(1.0, 3.0, n_samples)    # V2 shifts positive
        features[:, 2] -= rng.uniform(1.5, 3.5, n_samples)    # V3 shifts negative
        features[:, 3] += rng.uniform(0.5, 2.5, n_samples)    # V4 elevated
        features[:, 6] -= rng.uniform(1.0, 3.0, n_samples)    # V7 shifts negative
        features[:, 9] -= rng.uniform(1.0, 2.5, n_samples)    # V10 shifts negative
        features[:, 11] += rng.uniform(1.0, 3.0, n_samples)   # V12 elevated
        features[:, 13] -= rng.uniform(1.5, 4.0, n_samples)   # V14 shifts negative
        features[:, 15] -= rng.uniform(0.5, 2.0, n_samples)   # V16 shifts negative
        features[:, 16] -= rng.uniform(1.0, 2.5, n_samples)   # V17 shifts negative

    return features


def generate_transactions(n_total=100000, fraud_rate=0.02, seed=42):
    """
    Generate synthetic credit card transaction dataset.

    Args:
        n_total: Total number of transactions
        fraud_rate: Proportion of fraudulent transactions
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame with transaction data
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n_total * fraud_rate)
    n_legit = n_total - n_fraud

    print(f"Generating {n_total:,} transactions ({n_fraud:,} fraud, {n_legit:,} legitimate)...")

    # --- Generate legitimate transactions ---
    legit_features = generate_pca_features(n_legit, is_fraud=False, rng=rng)
    legit_amounts = np.abs(rng.lognormal(mean=3.5, sigma=1.2, size=n_legit))
    legit_amounts = np.clip(legit_amounts, 0.50, 5000.0)
    legit_times = np.sort(rng.uniform(0, 172800, n_legit))  # 48 hours in seconds
    legit_hour_p = np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08,
           0.08, 0.07, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04,
           0.03, 0.03, 0.02, 0.02])
    legit_hour_p /= legit_hour_p.sum()
    legit_hours = rng.choice(range(24), size=n_legit, p=legit_hour_p)
    legit_days = rng.integers(0, 7, n_legit)
    legit_categories = rng.choice(
        ['grocery', 'gas_station', 'restaurant', 'online', 'retail', 'travel', 'entertainment'],
        size=n_legit, p=[0.25, 0.12, 0.15, 0.20, 0.15, 0.05, 0.08]
    )

    # --- Generate fraudulent transactions ---
    fraud_features = generate_pca_features(n_fraud, is_fraud=True, rng=rng)
    fraud_amounts = np.abs(rng.lognormal(mean=5.0, sigma=1.5, size=n_fraud))
    fraud_amounts = np.clip(fraud_amounts, 1.0, 25000.0)
    fraud_times = rng.uniform(0, 172800, n_fraud)
    # Fraud skews towards late night / early morning
    fraud_hour_p = np.array([0.08, 0.09, 0.10, 0.10, 0.08, 0.06, 0.04, 0.03, 0.03, 0.03,
           0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02,
           0.02, 0.03, 0.04, 0.06])
    fraud_hour_p /= fraud_hour_p.sum()
    fraud_hours = rng.choice(range(24), size=n_fraud, p=fraud_hour_p)
    fraud_days = rng.integers(0, 7, n_fraud)
    fraud_categories = rng.choice(
        ['grocery', 'gas_station', 'restaurant', 'online', 'retail', 'travel', 'entertainment'],
        size=n_fraud, p=[0.05, 0.08, 0.05, 0.40, 0.15, 0.20, 0.07]
    )

    # --- Combine into DataFrame ---
    pca_columns = [f'V{i}' for i in range(1, 29)]

    df_legit = pd.DataFrame(legit_features, columns=pca_columns)
    df_legit['Amount'] = legit_amounts
    df_legit['Time'] = legit_times
    df_legit['TransactionHour'] = legit_hours
    df_legit['DayOfWeek'] = legit_days
    df_legit['MerchantCategory'] = legit_categories
    df_legit['Class'] = 0

    df_fraud = pd.DataFrame(fraud_features, columns=pca_columns)
    df_fraud['Amount'] = fraud_amounts
    df_fraud['Time'] = fraud_times
    df_fraud['TransactionHour'] = fraud_hours
    df_fraud['DayOfWeek'] = fraud_days
    df_fraud['MerchantCategory'] = fraud_categories
    df_fraud['Class'] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)

    # Assign synthetic card IDs (for LSTM sequence grouping)
    n_cards = 5000
    df['CardID'] = rng.integers(0, n_cards, len(df))

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def main():
    """Generate and save the dataset."""
    output_dir = Path(__file__).parent
    output_path = output_dir / 'transactions.csv'

    df = generate_transactions()

    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['Class'].value_counts().to_string())
    print(f"\nFraud rate: {df['Class'].mean():.2%}")
    print(f"\nSample statistics:")
    print(df[['Amount', 'Time', 'TransactionHour']].describe().to_string())


if __name__ == '__main__':
    main()
