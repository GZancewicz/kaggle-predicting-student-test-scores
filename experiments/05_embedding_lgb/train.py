"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 05: Neural Network Embeddings + LightGBM

Strategy:
1. Train MLP with learned categorical embeddings
2. Extract the learned embeddings
3. Use embeddings as features in LightGBM
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
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


class EmbeddingMLP(nn.Module):
    """MLP with learned categorical embeddings."""

    def __init__(self, cat_cardinalities, cat_emb_dims, num_features, hidden_dims=[128, 64, 32], dropout=0.2):
        """
        Args:
            cat_cardinalities: dict of {col_name: num_categories}
            cat_emb_dims: dict of {col_name: embedding_dim}
            num_features: number of numeric features
            hidden_dims: hidden layer sizes
            dropout: dropout rate
        """
        super().__init__()

        # Create embedding layers for each categorical
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, n_cats in cat_cardinalities.items():
            emb_dim = cat_emb_dims[col]
            self.embeddings[col] = nn.Embedding(n_cats, emb_dim)
            total_emb_dim += emb_dim

        # Input dim = numeric features + all embedding dims
        input_dim = num_features + total_emb_dim

        # Build MLP layers
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
        self.cat_cols = list(cat_cardinalities.keys())

    def forward(self, x_num, x_cat):
        """
        Args:
            x_num: numeric features (batch, num_features)
            x_cat: categorical indices (batch, n_cat_cols)
        """
        # Get embeddings for each categorical
        emb_list = []
        for i, col in enumerate(self.cat_cols):
            emb = self.embeddings[col](x_cat[:, i])
            emb_list.append(emb)

        # Concat numeric + all embeddings
        x = torch.cat([x_num] + emb_list, dim=1)
        return self.network(x).squeeze(-1)

    def get_embeddings(self):
        """Extract learned embedding weights."""
        return {col: emb.weight.data.cpu().numpy()
                for col, emb in self.embeddings.items()}


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def preprocess(train, test):
    """Separate numeric and categorical features."""
    train = train.copy()
    test = test.copy()

    # Numeric columns (truly continuous)
    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    # ALL categorical columns to embed (including ordinals)
    cat_cols = ['gender', 'course', 'study_method', 'sleep_quality',
                'internet_access', 'facility_rating', 'exam_difficulty']

    # Label encode categoricals
    label_encoders = {}
    cat_cardinalities = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        label_encoders[col] = le
        cat_cardinalities[col] = len(le.classes_)

    # Prepare arrays
    X_train_num = train[num_cols].values.astype(np.float32)
    X_train_cat = train[cat_cols].values.astype(np.int64)
    y_train = train['exam_score'].values.astype(np.float32)

    X_test_num = test[num_cols].values.astype(np.float32)
    X_test_cat = test[cat_cols].values.astype(np.int64)
    test_ids = test['id'].values

    return (X_train_num, X_train_cat, y_train,
            X_test_num, X_test_cat, test_ids,
            num_cols, cat_cols, cat_cardinalities, label_encoders)


def train_embedding_mlp(X_train_num, X_train_cat, y_train,
                        X_test_num, X_test_cat,
                        cat_cardinalities,
                        n_folds=5, epochs=100, batch_size=1024, lr=1e-3):
    """Train MLP with embeddings and extract learned embeddings."""
    print(f"\n{'='*50}")
    print("Training Embedding MLP")
    print('='*50)

    # Embedding dimensions (using rule of thumb: min(50, (n_cats + 1) // 2))
    cat_emb_dims = {col: min(50, (n + 1) // 2) for col, n in cat_cardinalities.items()}
    print(f"Embedding dimensions: {cat_emb_dims}")

    # Scale numeric features
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train_num))
    test_preds = np.zeros(len(X_test_num))
    all_embeddings = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_num)):
        print(f"\nFold {fold+1}/{n_folds}")

        X_tr_num = X_train_num_scaled[train_idx]
        X_tr_cat = X_train_cat[train_idx]
        y_tr = y_train[train_idx]

        X_val_num = X_train_num_scaled[val_idx]
        X_val_cat = X_train_cat[val_idx]
        y_val = y_train[val_idx]

        # DataLoaders
        train_ds = TensorDataset(
            torch.FloatTensor(X_tr_num),
            torch.LongTensor(X_tr_cat),
            torch.FloatTensor(y_tr)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val_num),
            torch.LongTensor(X_val_cat),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Model
        model = EmbeddingMLP(
            cat_cardinalities=cat_cardinalities,
            cat_emb_dims=cat_emb_dims,
            num_features=X_train_num.shape[1],
            hidden_dims=[128, 64, 32],
            dropout=0.2
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val_rmse = float('inf')
        best_model_state = None
        patience = 5
        no_improve = 0
        min_delta = 0.001  # Must improve by at least this much

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for X_num_batch, X_cat_batch, y_batch in train_loader:
                X_num_batch = X_num_batch.to(DEVICE)
                X_cat_batch = X_cat_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                pred = model(X_num_batch, X_cat_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
            train_loss /= len(train_ds)

            # Validate
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for X_num_batch, X_cat_batch, _ in val_loader:
                    X_num_batch = X_num_batch.to(DEVICE)
                    X_cat_batch = X_cat_batch.to(DEVICE)
                    pred = model(X_num_batch, X_cat_batch)
                    val_preds_list.append(pred.cpu().numpy())
            val_preds_arr = np.concatenate(val_preds_list)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds_arr))

            scheduler.step(val_rmse)

            if val_rmse < best_val_rmse - min_delta:
                best_val_rmse = val_rmse
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}, Best={best_val_rmse:.4f}")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)
        model.to(DEVICE)
        model.eval()

        # Store embeddings from this fold
        fold_embeddings = model.get_embeddings()
        all_embeddings.append(fold_embeddings)

        # OOF predictions
        with torch.no_grad():
            val_preds_list = []
            for X_num_batch, X_cat_batch, _ in val_loader:
                X_num_batch = X_num_batch.to(DEVICE)
                X_cat_batch = X_cat_batch.to(DEVICE)
                val_preds_list.append(model(X_num_batch, X_cat_batch).cpu().numpy())
            oof_preds[val_idx] = np.concatenate(val_preds_list)

        # Test predictions
        test_ds = TensorDataset(
            torch.FloatTensor(X_test_num_scaled),
            torch.LongTensor(X_test_cat),
            torch.zeros(len(X_test_num))
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            fold_test_preds = []
            for X_num_batch, X_cat_batch, _ in test_loader:
                X_num_batch = X_num_batch.to(DEVICE)
                X_cat_batch = X_cat_batch.to(DEVICE)
                fold_test_preds.append(model(X_num_batch, X_cat_batch).cpu().numpy())
            test_preds += np.concatenate(fold_test_preds) / n_folds

        print(f"  Fold {fold+1} Best RMSE: {best_val_rmse:.4f}")

    # Average embeddings across folds
    avg_embeddings = {}
    for col in all_embeddings[0].keys():
        avg_embeddings[col] = np.mean([e[col] for e in all_embeddings], axis=0)

    overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nEmbedding MLP Overall CV RMSE: {overall_rmse:.4f}")

    return oof_preds, test_preds, avg_embeddings, scaler, cat_emb_dims


def expand_with_embeddings(X_num, X_cat, embeddings, cat_cols):
    """Replace categorical indices with their learned embeddings."""
    expanded = [X_num]

    for i, col in enumerate(cat_cols):
        emb_matrix = embeddings[col]  # (n_categories, emb_dim)
        cat_indices = X_cat[:, i]
        cat_embs = emb_matrix[cat_indices]  # (n_samples, emb_dim)
        expanded.append(cat_embs)

    return np.hstack(expanded)


def train_lgb_with_embeddings(X_train_num, X_train_cat, y_train,
                              X_test_num, X_test_cat,
                              embeddings, cat_cols, scaler,
                              n_folds=5):
    """Train LightGBM using embedding-expanded features."""
    print(f"\n{'='*50}")
    print("Training LightGBM with Embedding Features")
    print('='*50)

    # Scale numeric and expand with embeddings
    X_train_num_scaled = scaler.transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train_expanded = expand_with_embeddings(X_train_num_scaled, X_train_cat, embeddings, cat_cols)
    X_test_expanded = expand_with_embeddings(X_test_num_scaled, X_test_cat, embeddings, cat_cols)

    print(f"Expanded feature dim: {X_train_expanded.shape[1]} (was {X_train_num.shape[1]} numeric + {X_train_cat.shape[1]} categorical)")

    # LightGBM params (from experiment 01)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 8,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 1000,
        'random_state': 42,
        'verbosity': -1
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train_expanded))
    test_preds = np.zeros(len(X_test_expanded))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_expanded)):
        print(f"\nFold {fold+1}/{n_folds}")

        X_tr, X_val = X_train_expanded[train_idx], X_train_expanded[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test_expanded) / n_folds

        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nLightGBM with Embeddings CV RMSE: {overall_rmse:.4f}")

    return oof_preds, test_preds


def create_submission(test_ids, predictions, filename):
    pd.DataFrame({'id': test_ids, 'exam_score': predictions}).to_csv(filename, index=False)
    print(f"Saved {filename}")


def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")

    print("\nPreprocessing...")
    (X_train_num, X_train_cat, y_train,
     X_test_num, X_test_cat, test_ids,
     num_cols, cat_cols, cat_cardinalities, label_encoders) = preprocess(train, test)

    print(f"Numeric features: {len(num_cols)}")
    print(f"Categorical features: {cat_cols}")
    print(f"Cardinalities: {cat_cardinalities}")

    # Step 1: Train embedding MLP
    mlp_oof, mlp_test, embeddings, scaler, cat_emb_dims = train_embedding_mlp(
        X_train_num, X_train_cat, y_train,
        X_test_num, X_test_cat,
        cat_cardinalities,
        n_folds=5, epochs=100, batch_size=1024, lr=1e-3
    )

    # Show learned embeddings
    print("\n" + "="*50)
    print("Learned Embeddings")
    print("="*50)
    for col, emb in embeddings.items():
        print(f"\n{col} embeddings ({emb.shape[0]} categories x {emb.shape[1]} dims):")
        le = label_encoders[col]
        for i, cat_name in enumerate(le.classes_):
            print(f"  {cat_name}: {emb[i].round(3)}")

    # Step 2: Train LightGBM with embedding features
    lgb_oof, lgb_test = train_lgb_with_embeddings(
        X_train_num, X_train_cat, y_train,
        X_test_num, X_test_cat,
        embeddings, cat_cols, scaler,
        n_folds=5
    )

    # Compare results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    mlp_rmse = np.sqrt(mean_squared_error(y_train, mlp_oof))
    lgb_rmse = np.sqrt(mean_squared_error(y_train, lgb_oof))
    print(f"Embedding MLP alone:      {mlp_rmse:.4f}")
    print(f"LightGBM with embeddings: {lgb_rmse:.4f}")

    # Ensemble
    for w in [0.3, 0.5, 0.7]:
        ens_oof = w * mlp_oof + (1 - w) * lgb_oof
        ens_rmse = np.sqrt(mean_squared_error(y_train, ens_oof))
        print(f"Ensemble (MLP={w:.1f}, LGB={1-w:.1f}): {ens_rmse:.4f}")

    # Save OOF predictions
    np.save('oof_lgb_emb.npy', lgb_oof)
    np.save('oof_mlp_emb.npy', mlp_oof)
    np.save('y_train.npy', y_train)
    print("Saved OOF predictions (.npy files)")

    # Save best submission (likely LGB with embeddings)
    create_submission(test_ids, lgb_test, 'submission_lgb_embeddings.csv')
    create_submission(test_ids, mlp_test, 'submission_mlp_embeddings.csv')

    # Ensemble submission
    best_ens = 0.3 * mlp_test + 0.7 * lgb_test
    create_submission(test_ids, best_ens, 'submission_ensemble.csv')


if __name__ == '__main__':
    main()
