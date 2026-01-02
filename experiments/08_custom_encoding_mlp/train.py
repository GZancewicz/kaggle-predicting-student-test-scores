"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 08: Custom Encoding MLP

Custom categorical encodings:
- gender: female=0, other=2, male=3
- course: frequency encoding (% of total)
- internet_access: no=0, yes=1
- sleep_quality: poor=0, average=1, good=3
- study_method: frequency encoding (% of total)
- facility_rating: low=0, medium=1, high=2 (standard ordinal)
- exam_difficulty: easy=0, moderate=1, hard=2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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


def compute_frequency_encoding(train_col, test_col):
    """Compute frequency encoding (% of total) from training data."""
    freq = train_col.value_counts(normalize=True).to_dict()
    train_encoded = train_col.map(freq)
    test_encoded = test_col.map(freq).fillna(0)  # Handle unseen categories
    return train_encoded, test_encoded


def preprocess(train, test):
    """Apply custom encodings."""
    train = train.copy()
    test = test.copy()

    # ========== CUSTOM CATEGORICAL ENCODINGS ==========

    # gender: female=0, other=2, male=3
    gender_map = {'female': 0, 'other': 2, 'male': 3}
    train['gender_enc'] = train['gender'].map(gender_map)
    test['gender_enc'] = test['gender'].map(gender_map)
    print(f"gender encoding: {gender_map}")

    # course: frequency encoding (% of total)
    train['course_enc'], test['course_enc'] = compute_frequency_encoding(train['course'], test['course'])
    print(f"course frequency encoding applied")

    # internet_access: no=0, yes=1
    internet_map = {'no': 0, 'yes': 1}
    train['internet_enc'] = train['internet_access'].map(internet_map)
    test['internet_enc'] = test['internet_access'].map(internet_map)
    print(f"internet_access encoding: {internet_map}")

    # sleep_quality: poor=0, average=1, good=3
    sleep_map = {'poor': 0, 'average': 1, 'good': 3}
    train['sleep_quality_enc'] = train['sleep_quality'].map(sleep_map)
    test['sleep_quality_enc'] = test['sleep_quality'].map(sleep_map)
    print(f"sleep_quality encoding: {sleep_map}")

    # study_method: frequency encoding (% of total)
    train['study_method_enc'], test['study_method_enc'] = compute_frequency_encoding(train['study_method'], test['study_method'])
    print(f"study_method frequency encoding applied")

    # facility_rating: low=0, medium=1, high=2 (standard ordinal)
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    train['facility_enc'] = train['facility_rating'].map(facility_map)
    test['facility_enc'] = test['facility_rating'].map(facility_map)
    print(f"facility_rating encoding: {facility_map}")

    # exam_difficulty: easy=0, moderate=1, hard=2
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    train['difficulty_enc'] = train['exam_difficulty'].map(difficulty_map)
    test['difficulty_enc'] = test['exam_difficulty'].map(difficulty_map)
    print(f"exam_difficulty encoding: {difficulty_map}")

    # ========== ENGINEERED FEATURES ==========

    # From residual analysis
    train['study_hours_sq'] = train['study_hours'] ** 2
    test['study_hours_sq'] = test['study_hours'] ** 2

    train['study_hours_x_difficulty'] = train['study_hours'] * train['difficulty_enc']
    test['study_hours_x_difficulty'] = test['study_hours'] * test['difficulty_enc']

    train['sleep_score'] = train['sleep_hours'] * train['sleep_quality_enc']
    test['sleep_score'] = test['sleep_hours'] * test['sleep_quality_enc']

    train['resource_score'] = train['facility_enc'] * train['internet_enc']
    test['resource_score'] = test['facility_enc'] * test['internet_enc']

    train['study_efficiency'] = train['study_hours'] * train['class_attendance'] / 100
    test['study_efficiency'] = test['study_hours'] * test['class_attendance'] / 100

    train['low_attendance'] = (train['class_attendance'] < 60).astype(int)
    test['low_attendance'] = (test['class_attendance'] < 60).astype(int)

    # ========== FEATURE COLUMNS ==========

    feature_cols = [
        # Base numeric
        'age', 'study_hours', 'class_attendance', 'sleep_hours',
        # Custom encoded categoricals
        'gender_enc', 'course_enc', 'internet_enc', 'sleep_quality_enc',
        'study_method_enc', 'facility_enc', 'difficulty_enc',
        # Engineered features
        'study_hours_sq', 'study_hours_x_difficulty', 'sleep_score',
        'resource_score', 'study_efficiency', 'low_attendance'
    ]

    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train['exam_score'].values.astype(np.float32)
    X_test = test[feature_cols].values.astype(np.float32)
    test_ids = test['id'].values

    return X_train, y_train, X_test, test_ids, feature_cols


def train_mlp(X_train, y_train, X_test, n_folds=5, epochs=100, batch_size=1024, lr=1e-3):
    """Train MLP with cross-validation."""
    print(f"\n{'='*50}")
    print("Training MLP with Custom Encodings")
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
            val_preds_list = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    pred = model(X_batch)
                    val_preds_list.append(pred.cpu().numpy())
            val_preds_arr = np.concatenate(val_preds_list)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds_arr))

            scheduler.step(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or no_improve == 0:
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}, Best={best_val_rmse:.4f}")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)
        model.eval()

        # OOF predictions
        with torch.no_grad():
            val_preds_list = []
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(DEVICE)
                val_preds_list.append(model(X_batch).cpu().numpy())
            oof_preds[val_idx] = np.concatenate(val_preds_list)

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

    print("\nPreprocessing with custom encodings...")
    X_train, y_train, X_test, test_ids, feature_cols = preprocess(train, test)
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    # Train MLP
    mlp_oof, mlp_test = train_mlp(X_train, y_train, X_test, n_folds=5, epochs=100, batch_size=1024, lr=1e-3)

    # Compare to baselines
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    mlp_rmse = np.sqrt(mean_squared_error(y_train, mlp_oof))
    print(f"Exp 08 Custom Encoding MLP: {mlp_rmse:.4f}")

    try:
        exp05_oof = np.load('../05_embedding_lgb/oof_lgb_emb.npy')
        exp05_rmse = np.sqrt(mean_squared_error(y_train, exp05_oof))
        print(f"Exp 05 Embedding LGB:       {exp05_rmse:.4f}")
        print(f"Difference:                 {mlp_rmse - exp05_rmse:+.4f}")
    except:
        pass

    # Save OOF predictions
    np.save('oof_mlp.npy', mlp_oof)
    np.save('y_train.npy', y_train)
    print("\nSaved OOF predictions (.npy files)")

    # Save submission
    create_submission(test_ids, mlp_test, 'submission_mlp.csv')


if __name__ == '__main__':
    main()
