"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 10: LGB/XGB Ensemble with Polynomial Features

Combines:
- Exp 01's LGB/XGB ensemble approach
- Exp 09's polynomial features (custom encodings + squared + cross-terms)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def compute_frequency_encoding(train_col, test_col):
    """Compute frequency encoding (% of total) from training data."""
    freq = train_col.value_counts(normalize=True).to_dict()
    train_encoded = train_col.map(freq)
    test_encoded = test_col.map(freq).fillna(0)
    return train_encoded, test_encoded


def preprocess(train, test):
    """Apply custom encodings + polynomial features (same as Exp 09)."""
    train = train.copy()
    test = test.copy()

    # ========== CUSTOM CATEGORICAL ENCODINGS (from Exp 08/09) ==========

    # gender: female=0, other=2, male=3
    gender_map = {'female': 0, 'other': 2, 'male': 3}
    train['gender_enc'] = train['gender'].map(gender_map)
    test['gender_enc'] = test['gender'].map(gender_map)

    # course: frequency encoding
    train['course_enc'], test['course_enc'] = compute_frequency_encoding(train['course'], test['course'])

    # internet_access: no=0, yes=1
    internet_map = {'no': 0, 'yes': 1}
    train['internet_enc'] = train['internet_access'].map(internet_map)
    test['internet_enc'] = test['internet_access'].map(internet_map)

    # sleep_quality: poor=0, average=1, good=3
    sleep_map = {'poor': 0, 'average': 1, 'good': 3}
    train['sleep_quality_enc'] = train['sleep_quality'].map(sleep_map)
    test['sleep_quality_enc'] = test['sleep_quality'].map(sleep_map)

    # study_method: frequency encoding
    train['study_method_enc'], test['study_method_enc'] = compute_frequency_encoding(train['study_method'], test['study_method'])

    # facility_rating: low=0, medium=1, high=2
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    train['facility_enc'] = train['facility_rating'].map(facility_map)
    test['facility_enc'] = test['facility_rating'].map(facility_map)

    # exam_difficulty: easy=0, moderate=1, hard=2
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    train['difficulty_enc'] = train['exam_difficulty'].map(difficulty_map)
    test['difficulty_enc'] = test['exam_difficulty'].map(difficulty_map)

    # ========== BASE FEATURE COLUMNS ==========

    base_cols = [
        # Base numeric
        'age', 'study_hours', 'class_attendance', 'sleep_hours',
        # Custom encoded categoricals
        'gender_enc', 'course_enc', 'internet_enc', 'sleep_quality_enc',
        'study_method_enc', 'facility_enc', 'difficulty_enc',
    ]

    X_train_base = train[base_cols].values.astype(np.float32)
    X_test_base = test[base_cols].values.astype(np.float32)

    # ========== POLYNOMIAL EXPANSION ==========

    print(f"Base features: {len(base_cols)}")

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train_base).astype(np.float32)
    X_test_poly = poly.transform(X_test_base).astype(np.float32)

    feature_names = poly.get_feature_names_out(base_cols)
    print(f"After polynomial expansion: {len(feature_names)} features")

    y_train = train['exam_score'].values.astype(np.float32)
    test_ids = test['id'].values

    return X_train_poly, y_train, X_test_poly, test_ids, feature_names


def train_lightgbm(X_train, y_train, X_test, n_folds=5):
    """LightGBM with hand-tuned params."""
    print("\n" + "="*50)
    print("Training LightGBM")
    print("="*50)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params, train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds
        print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, oof_preds[val_idx])):.4f}")

    print(f"\nLightGBM CV RMSE: {np.sqrt(mean_squared_error(y_train, oof_preds)):.4f}")
    return oof_preds, test_preds


def train_xgboost(X_train, y_train, X_test, n_folds=5):
    """XGBoost with hand-tuned params."""
    print("\n" + "="*50)
    print("Training XGBoost")
    print("="*50)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_child_weight': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'seed': 42
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params, dtrain,
            num_boost_round=2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=200
        )

        oof_preds[val_idx] = model.predict(dval)
        test_preds += model.predict(xgb.DMatrix(X_test)) / n_folds
        print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, oof_preds[val_idx])):.4f}")

    print(f"\nXGBoost CV RMSE: {np.sqrt(mean_squared_error(y_train, oof_preds)):.4f}")
    return oof_preds, test_preds


def optimize_weights(oof_list, y_train):
    """Find optimal blend weights."""
    def rmse_loss(weights):
        weights = np.array(weights) / np.sum(weights)
        blended = sum(w * p for w, p in zip(weights, oof_list))
        return np.sqrt(mean_squared_error(y_train, blended))

    n = len(oof_list)
    result = minimize(rmse_loss, [1/n]*n, bounds=[(0,1)]*n, method='SLSQP')
    weights = np.array(result.x) / np.sum(result.x)
    return weights


def create_submission(test_ids, predictions, filename):
    pd.DataFrame({'id': test_ids, 'exam_score': predictions}).to_csv(filename, index=False)
    print(f"Saved {filename}")


def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")

    print("\nPreprocessing with polynomial features...")
    X_train, y_train, X_test, test_ids, feature_names = preprocess(train, test)

    # Train models
    lgb_oof, lgb_test = train_lightgbm(X_train, y_train, X_test)
    xgb_oof, xgb_test = train_xgboost(X_train, y_train, X_test)

    # Ensemble
    print("\n" + "="*50)
    print("Creating Ensemble")
    print("="*50)

    oof_list = [lgb_oof, xgb_oof]
    test_list = [lgb_test, xgb_test]
    names = ['LGB', 'XGB']

    weights = optimize_weights(oof_list, y_train)
    print(f"Optimal weights: {dict(zip(names, weights.round(3)))}")

    final_oof = sum(w * p for w, p in zip(weights, oof_list))
    final_test = sum(w * p for w, p in zip(weights, test_list))

    ensemble_rmse = np.sqrt(mean_squared_error(y_train, final_oof))
    print(f"\nFinal Ensemble CV RMSE: {ensemble_rmse:.4f}")

    # Compare to baselines
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Exp 10 Ensemble (poly features): {ensemble_rmse:.4f}")
    print(f"Exp 01 Ensemble (original):      8.7411")
    print(f"Exp 07 LGB Enhanced:             8.7395")
    print(f"Difference vs Exp 01:            {ensemble_rmse - 8.7411:+.4f}")

    # Save OOF predictions
    np.save('oof_lgb.npy', lgb_oof)
    np.save('oof_xgb.npy', xgb_oof)
    np.save('oof_ensemble.npy', final_oof)
    np.save('y_train.npy', y_train)
    print("\nSaved OOF predictions (.npy files)")

    # Save submissions
    create_submission(test_ids, lgb_test, 'submission_lgb.csv')
    create_submission(test_ids, xgb_test, 'submission_xgb.csv')
    create_submission(test_ids, final_test, 'submission_ensemble.csv')


if __name__ == '__main__':
    main()
