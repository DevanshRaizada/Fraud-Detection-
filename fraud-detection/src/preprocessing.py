"""
Data Preprocessing & Feature Engineering Pipeline
===================================================
Handles data loading, cleaning, feature engineering, scaling,
train/test splitting with stratification, and SMOTE oversampling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')


class FraudPreprocessor:
    """End-to-end preprocessing pipeline for fraud detection."""

    def __init__(self, data_path=None):
        self.data_path = data_path or Path(__file__).parent.parent / 'data' / 'transactions.csv'
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.models_dir = Path(__file__).parent.parent / 'models'
        self.models_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load transaction dataset."""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(df):,} transactions")
        print(f"  Fraud rate: {df['Class'].mean():.2%}")
        return df

    def engineer_features(self, df):
        """Create engineered features from raw data."""
        print("Engineering features...")
        df = df.copy()

        # Encode merchant category
        df['MerchantCategory_encoded'] = self.label_encoder.fit_transform(df['MerchantCategory'])

        # Cyclical encoding for hour of day
        df['Hour_sin'] = np.sin(2 * np.pi * df['TransactionHour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['TransactionHour'] / 24)

        # Cyclical encoding for day of week
        df['Day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['Day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        # Log-transform amount (reduces skew)
        df['Amount_log'] = np.log1p(df['Amount'])

        # Amount bins
        df['Amount_bin'] = pd.cut(df['Amount'],
                                   bins=[0, 10, 50, 200, 1000, 50000],
                                   labels=[0, 1, 2, 3, 4]).astype(float)

        # Is weekend flag
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

        # Is night flag (10pm - 6am)
        df['IsNight'] = ((df['TransactionHour'] >= 22) | (df['TransactionHour'] <= 6)).astype(int)

        return df

    def get_feature_columns(self, df):
        """Define feature columns for tree-based models."""
        pca_cols = [f'V{i}' for i in range(1, 29)]
        engineered_cols = [
            'Amount', 'Amount_log', 'Amount_bin',
            'TransactionHour', 'DayOfWeek',
            'MerchantCategory_encoded',
            'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos',
            'IsWeekend', 'IsNight', 'Time'
        ]
        self.feature_columns = pca_cols + engineered_cols
        return self.feature_columns

    def prepare_tree_data(self, test_size=0.2, apply_smote=True):
        """
        Prepare data for tree-based models (XGBoost, LightGBM).

        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        df = self.load_data()
        df = self.engineer_features(df)
        features = self.get_feature_columns(df)

        X = df[features].values
        y = df['Class'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nTrain set: {len(X_train):,} samples (fraud: {y_train.sum():,})")
        print(f"Test set:  {len(X_test):,} samples (fraud: {y_test.sum():,})")

        if apply_smote:
            print("Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42, sampling_strategy=0.3)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {len(X_train):,} samples (fraud: {y_train.sum():,})")

        # Save scaler
        joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, self.models_dir / 'label_encoder.pkl')
        joblib.dump(features, self.models_dir / 'feature_columns.pkl')
        print(f"Saved scaler and encoder to {self.models_dir}")

        return X_train, X_test, y_train, y_test, features

    def prepare_lstm_data(self, sequence_length=10, test_size=0.2):
        """
        Prepare sequential data for LSTM model.
        Groups transactions by CardID and creates sliding windows.

        Returns:
            X_train_seq, X_test_seq, y_train_seq, y_test_seq
        """
        df = self.load_data()
        df = self.engineer_features(df)
        features = self.get_feature_columns(df)

        # Sort by CardID and Time for sequential ordering
        df = df.sort_values(['CardID', 'Time']).reset_index(drop=True)

        X_scaled = self.scaler.fit_transform(df[features].values)
        df_scaled = pd.DataFrame(X_scaled, columns=features)
        df_scaled['Class'] = df['Class'].values
        df_scaled['CardID'] = df['CardID'].values

        sequences = []
        labels = []

        print(f"Creating sequences (window={sequence_length})...")

        for card_id, group in df_scaled.groupby('CardID'):
            if len(group) < sequence_length:
                continue

            card_features = group[features].values
            card_labels = group['Class'].values

            for i in range(sequence_length, len(group)):
                seq = card_features[i - sequence_length:i]
                sequences.append(seq)
                labels.append(card_labels[i])

        X_seq = np.array(sequences)
        y_seq = np.array(labels)

        print(f"Generated {len(X_seq):,} sequences")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, random_state=42, stratify=y_seq
        )

        print(f"LSTM Train: {len(X_train):,} sequences (fraud: {y_train.sum():,})")
        print(f"LSTM Test:  {len(X_test):,} sequences (fraud: {y_test.sum():,})")

        return X_train, X_test, y_train, y_test


def main():
    """Test preprocessing pipeline."""
    preprocessor = FraudPreprocessor()

    print("=" * 60)
    print("TREE-BASED MODEL DATA")
    print("=" * 60)
    X_train, X_test, y_train, y_test, features = preprocessor.prepare_tree_data()
    print(f"\nFeature count: {len(features)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print("\n" + "=" * 60)
    print("LSTM SEQUENTIAL DATA")
    print("=" * 60)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = preprocessor.prepare_lstm_data()
    print(f"\nX_train_seq shape: {X_train_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}")


if __name__ == '__main__':
    main()
