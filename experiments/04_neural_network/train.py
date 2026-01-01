"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 04: Simple MLP Neural Network
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def preprocess(train, test):
    """Encode all features as numeric."""
    train = train.copy()
    test = test.copy()

    # Ordinal encodings
    sleep_quality_map = {'poor': 1, 'average': 2, 'good': 3}
    internet_map = {'no': 0, 'yes': 1}
    facility_map = {'low': 1, 'medium': 2, 'high': 3}
    difficulty_map = {'easy': 1, 'moderate': 2, 'hard': 3}

    for df in [train, test]:
        df['sleep_quality'] = df['sleep_quality'].map(sleep_quality_map)
        df['internet_access'] = df['internet_access'].map(internet_map)
        df['facility_rating'] = df['facility_rating'].map(facility_map)
        df['exam_difficulty'] = df['exam_difficulty'].map(difficulty_map)

    # Label encode nominal categoricals
    cat_cols = ['gender', 'course', 'study_method']
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # Feature columns
    feature_cols = ['age', 'gender', 'course', 'study_hours', 'class_attendance',
                    'internet_access', 'sleep_hours', 'sleep_quality',
                    'study_method', 'facility_rating', 'exam_difficulty']

    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train['exam_score'].values.astype(np.float32)
    X_test = test[feature_cols].values.astype(np.float32)
    test_ids = test['id'].values

    return X_train, y_train, X_test, test_ids, feature_cols


def train_mlp(X_train, y_train, X_test, n_folds=5, epochs=50, batch_size=1024, lr=1e-3):
    """Train MLP with cross-validation."""
    print(f"\n{'='*50}")
    print("Training MLP")
    print('='*50)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        print(f"\nFold {fold+1}/{n_folds}")

        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # DataLoaders
        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_ds = TensorDataset(torch.FloatTensor(X_test_scaled), torch.zeros(len(X_test_scaled)))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Model
        model = MLP(input_dim=X_train.shape[1], hidden_dims=[128, 64, 32], dropout=0.2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val_rmse = float('inf')
        best_model_state = None
        patience = 10
        no_improve = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
            train_loss /= len(train_ds)

            # Validate
            model.eval()
            val_preds = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    pred = model(X_batch)
                    val_preds.append(pred.cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

            scheduler.step(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}, Best={best_val_rmse:.4f}")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)
        model.eval()

        # OOF predictions
        with torch.no_grad():
            val_preds = []
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(DEVICE)
                val_preds.append(model(X_batch).cpu().numpy())
            oof_preds[val_idx] = np.concatenate(val_preds)

        # Test predictions
        with torch.no_grad():
            fold_test_preds = []
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(DEVICE)
                fold_test_preds.append(model(X_batch).cpu().numpy())
            test_preds += np.concatenate(fold_test_preds) / n_folds

        print(f"  Fold {fold+1} Best RMSE: {best_val_rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nMLP Overall CV RMSE: {overall_rmse:.4f}")

    return oof_preds, test_preds


def create_submission(test_ids, predictions, filename):
    pd.DataFrame({'id': test_ids, 'exam_score': predictions}).to_csv(filename, index=False)
    print(f"Saved {filename}")


def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")

    print("\nPreprocessing...")
    X_train, y_train, X_test, test_ids, feature_cols = preprocess(train, test)
    print(f"Features: {len(feature_cols)}")

    # Train MLP
    mlp_oof, mlp_test = train_mlp(X_train, y_train, X_test, n_folds=5, epochs=100, batch_size=1024, lr=1e-3)

    # Save OOF predictions
    np.save('oof_mlp.npy', mlp_oof)
    np.save('y_train.npy', y_train)
    print("Saved OOF predictions (.npy files)")

    # Save submission
    create_submission(test_ids, mlp_test, 'submission_mlp.csv')


if __name__ == '__main__':
    main()
