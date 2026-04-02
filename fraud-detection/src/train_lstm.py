"""
LSTM Fraud Detection Model
============================
Deep learning model using LSTM layers to capture sequential
spending patterns for fraud detection. Mirrors the approach
used by Credit company for temporal pattern recognition.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import FraudPreprocessor


class FraudLSTM(nn.Module):
    """LSTM network for sequential fraud detection."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(FraudLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass: LSTM -> last hidden state -> classifier."""
        lstm_out, (h_n, _) = self.lstm(x)
        # Use the last hidden state from the final LSTM layer
        last_hidden = h_n[-1]
        output = self.classifier(last_hidden)
        return output.squeeze(-1)


def train_lstm(epochs=20, batch_size=256, lr=0.001):
    """Train LSTM fraud detection model."""
    print("=" * 60)
    print("LSTM FRAUD DETECTION MODEL")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare sequential data
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_lstm_data(sequence_length=10)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = FraudLSTM(input_size=input_size, hidden_size=64, num_layers=2).to(device)

    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Class weighting for imbalanced data
    n_fraud = y_train.sum()
    n_legit = len(y_train) - n_fraud
    pos_weight = torch.tensor([n_legit / max(n_fraud, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Override: use BCE since we have sigmoid in model
    criterion = nn.BCELoss(weight=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    best_auc = 0.0
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'lstm_fraud.pt'

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t)
            y_prob = test_outputs.cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)
            auc = roc_auc_score(y_test, y_prob)

        scheduler.step(test_loss)

        avg_loss = epoch_loss / n_batches
        curr_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Val Loss: {test_loss:.4f} | "
                  f"AUC: {auc:.4f} | "
                  f"LR: {curr_lr:.6f}")

        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_size': 64,
                'num_layers': 2,
                'best_auc': best_auc,
            }, model_path)

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f}s")
    print(f"Best AUC-ROC: {best_auc:.4f}")

    # Load best model and final evaluation
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        y_prob = model(X_test_t).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"\nModel saved to: {model_path}")

    return model, auc_roc, train_time


if __name__ == '__main__':
    train_lstm()
