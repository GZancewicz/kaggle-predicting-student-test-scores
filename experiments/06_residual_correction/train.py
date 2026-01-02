"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 06: Post-prediction residual correction on Experiment 05

Idea: Use the residual slopes from Exp 05 to apply a linear correction
to predictions based on feature values.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def load_exp05_predictions():
    """Load OOF and test predictions from experiment 05."""
    oof_preds = np.load('../05_embedding_lgb/oof_lgb_emb.npy')
    y_train = np.load('../05_embedding_lgb/y_train.npy')

    # Load test predictions
    test_preds = pd.read_csv('../05_embedding_lgb/submission_lgb_embeddings.csv')['exam_score'].values

    return oof_preds, y_train, test_preds


def compute_residual_slopes(train_df, residuals):
    """Compute residual slopes for each numeric feature."""
    numeric_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    slopes = {}
    for col in numeric_cols:
        slope, intercept, _, p_value, _ = stats.linregress(train_df[col], residuals)
        slopes[col] = {'slope': slope, 'intercept': intercept, 'p_value': p_value}
        print(f"{col}: slope={slope:.4f}, p={p_value:.4f}")

    return slopes


def apply_linear_correction(preds, df, slopes):
    """Apply linear correction based on feature slopes."""
    corrected = preds.copy()

    for col, params in slopes.items():
        # Only apply if slope is significant
        if params['p_value'] < 0.05:
            correction = params['slope'] * df[col].values
            corrected += correction

    return corrected


def train_residual_model(train_df, residuals, test_df, n_folds=5):
    """Train a model to predict residuals (stacking approach)."""
    print("\n" + "="*50)
    print("Training Residual Correction Model")
    print("="*50)

    # Features for residual prediction
    feature_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_residual = residuals

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_corrections = np.zeros(len(X_train))
    test_corrections = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_residual[train_idx], y_residual[val_idx]

        # Simple Ridge regression to predict residuals
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)

        oof_corrections[val_idx] = model.predict(X_val)
        test_corrections += model.predict(X_test) / n_folds

        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_corrections[val_idx]))
        print(f"Fold {fold+1} Residual RMSE: {fold_rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_residual, oof_corrections))
    print(f"\nResidual Model CV RMSE: {overall_rmse:.4f}")

    return oof_corrections, test_corrections


def create_submission(test_ids, predictions, filename):
    pd.DataFrame({'id': test_ids, 'exam_score': predictions}).to_csv(filename, index=False)
    print(f"Saved {filename}")


def main():
    print("Loading data...")
    train_df, test_df = load_data()
    test_ids = test_df['id'].values
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    print("\nLoading Experiment 05 predictions...")
    oof_preds, y_train, test_preds = load_exp05_predictions()

    # Compute residuals
    residuals = y_train - oof_preds
    baseline_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"Experiment 05 CV RMSE: {baseline_rmse:.4f}")

    # Method 1: Simple linear correction
    print("\n" + "="*50)
    print("Method 1: Simple Linear Correction")
    print("="*50)
    slopes = compute_residual_slopes(train_df, residuals)

    corrected_oof = apply_linear_correction(oof_preds, train_df, slopes)
    corrected_test = apply_linear_correction(test_preds, test_df, slopes)

    corrected_rmse = np.sqrt(mean_squared_error(y_train, corrected_oof))
    print(f"\nAfter Linear Correction CV RMSE: {corrected_rmse:.4f}")
    print(f"Improvement: {baseline_rmse - corrected_rmse:.4f}")

    # Method 2: Train a residual model (stacking)
    oof_corrections, test_corrections = train_residual_model(train_df, residuals, test_df)

    stacked_oof = oof_preds + oof_corrections
    stacked_test = test_preds + test_corrections

    stacked_rmse = np.sqrt(mean_squared_error(y_train, stacked_oof))
    print(f"\nAfter Stacked Correction CV RMSE: {stacked_rmse:.4f}")
    print(f"Improvement: {baseline_rmse - stacked_rmse:.4f}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Baseline (Exp 05):        {baseline_rmse:.4f}")
    print(f"+ Linear Correction:      {corrected_rmse:.4f} ({baseline_rmse - corrected_rmse:+.4f})")
    print(f"+ Stacked Correction:     {stacked_rmse:.4f} ({baseline_rmse - stacked_rmse:+.4f})")

    # Save best result
    if stacked_rmse < corrected_rmse:
        best_oof = stacked_oof
        best_test = stacked_test
        best_name = "stacked"
    else:
        best_oof = corrected_oof
        best_test = corrected_test
        best_name = "linear"

    # Save OOF predictions
    np.save('oof_corrected.npy', best_oof)
    np.save('y_train.npy', y_train)
    print(f"\nSaved OOF predictions (best={best_name})")

    # Save submission
    create_submission(test_ids, best_test, 'submission_corrected.csv')


if __name__ == '__main__':
    main()
