"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 11: Stacking Ensemble with Target Encoding

Components:
1. CatBoost with native categoricals (3 seeds, scalable to 10)
2. LightGBM with OOF target encoding + frequency encoding (GOSS, 3 seeds)
3. Ridge stacker on OOF predictions
4. Linear calibration + clipping

Based on GPT recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import lightgbm as lgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_FOLDS = 5
CATBOOST_SEEDS = [42]  # Start with 1, then scale to [42, 202, 999]
LGB_SEEDS = [42]       # Start with 1, then scale to [42, 202, 999]
TARGET_SMOOTHING_VALUES = [10]  # Can try [5, 10, 20, 50]
CLIP_MIN, CLIP_MAX = 19, 100  # Target range


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def get_fold_indices(n_samples, n_folds=5, random_state=42):
    """Get consistent fold indices for all models."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


def target_encode_oof(train_col, y, fold_indices, m=10, noise=0):
    """
    OOF-safe target encoding with smoothing.
    enc = (sum_y + m * global_mean) / (count + m)
    """
    global_mean = y.mean()
    encoded = np.zeros(len(train_col))

    for train_idx, val_idx in fold_indices:
        # Compute encoding from training fold only
        df_fold = pd.DataFrame({'cat': train_col.iloc[train_idx], 'y': y[train_idx]})
        stats = df_fold.groupby('cat')['y'].agg(['sum', 'count'])
        stats['enc'] = (stats['sum'] + m * global_mean) / (stats['count'] + m)

        # Apply to validation fold
        encoded[val_idx] = train_col.iloc[val_idx].map(stats['enc']).fillna(global_mean).values

    # Add tiny noise to reduce overfit
    if noise > 0:
        encoded += np.random.normal(0, noise, len(encoded))

    return encoded


def target_encode_test(train_col, y, test_col, m=10):
    """Target encoding for test set using full training data."""
    global_mean = y.mean()
    df = pd.DataFrame({'cat': train_col, 'y': y})
    stats = df.groupby('cat')['y'].agg(['sum', 'count'])
    stats['enc'] = (stats['sum'] + m * global_mean) / (stats['count'] + m)
    return test_col.map(stats['enc']).fillna(global_mean).values


def frequency_encode(train_col, test_col):
    """Frequency encoding (count-based)."""
    freq = train_col.value_counts().to_dict()
    train_enc = train_col.map(freq).values
    test_enc = test_col.map(freq).fillna(0).values
    return train_enc, test_enc


def preprocess_catboost(train, test):
    """Prepare data for CatBoost with native categoricals."""
    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']
    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    # Keep categoricals as strings for CatBoost
    X_train = train[num_cols + cat_cols].copy()
    X_test = test[num_cols + cat_cols].copy()

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    y_train = train['exam_score'].values
    test_ids = test['id'].values

    return X_train, y_train, X_test, test_ids, cat_cols, num_cols


def preprocess_lgb_target_encoded(train, test, y_train, fold_indices, m=10):
    """Prepare data for LightGBM with target + frequency encoding."""
    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']
    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    X_train = train[num_cols].copy()
    X_test = test[num_cols].copy()

    # Add target encoding and frequency encoding for each categorical
    for col in cat_cols:
        # Target encoding (OOF for train, full for test)
        X_train[f'{col}_te'] = target_encode_oof(train[col], y_train, fold_indices, m=m)
        X_test[f'{col}_te'] = target_encode_test(train[col], y_train, test[col], m=m)

        # Frequency encoding
        train_freq, test_freq = frequency_encode(train[col], test[col])
        X_train[f'{col}_freq'] = train_freq
        X_test[f'{col}_freq'] = test_freq

    return X_train.values.astype(np.float32), X_test.values.astype(np.float32)


def train_catboost(X_train, y_train, X_test, cat_cols, fold_indices, seeds):
    """Train CatBoost with native categoricals, multiple seeds."""
    print("\n" + "="*60)
    print(f"Training CatBoost ({len(seeds)} seeds)")
    print("="*60)

    cat_features = [X_train.columns.get_loc(c) for c in cat_cols]

    all_oof = []
    all_test = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]

            model = CatBoostRegressor(
                iterations=2000,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=3,
                loss_function='RMSE',
                eval_metric='RMSE',
                random_seed=seed,
                boosting_type='Ordered',  # Ordered boosting for better categorical handling
                od_type='Iter',
                od_wait=100,
                verbose=200,
                cat_features=cat_features
            )

            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / len(fold_indices)

            fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
            print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

        seed_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
        print(f"Seed {seed} CV RMSE: {seed_rmse:.4f}")

        all_oof.append(oof_preds)
        all_test.append(test_preds)

        # Save per-seed OOF
        np.save(f'oof_catboost_seed{seed}.npy', oof_preds)

    # Average across seeds
    mean_oof = np.mean(all_oof, axis=0)
    mean_test = np.mean(all_test, axis=0)

    mean_rmse = np.sqrt(mean_squared_error(y_train, mean_oof))
    print(f"\nCatBoost Mean CV RMSE: {mean_rmse:.4f}")

    np.save('oof_catboost_mean.npy', mean_oof)

    return mean_oof, mean_test, all_oof, all_test


def train_lgb_target_encoded(X_train, y_train, X_test, fold_indices, seeds):
    """Train LightGBM with GOSS on target-encoded features, multiple seeds."""
    print("\n" + "="*60)
    print(f"Training LightGBM GOSS ({len(seeds)} seeds)")
    print("="*60)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'goss',  # GOSS for speed
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 30,
        'feature_fraction': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
    }

    all_oof = []
    all_test = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        params['seed'] = seed

        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(fold_indices):
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
            test_preds += model.predict(X_test) / len(fold_indices)

            fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
            print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

        seed_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
        print(f"Seed {seed} CV RMSE: {seed_rmse:.4f}")

        all_oof.append(oof_preds)
        all_test.append(test_preds)

        np.save(f'oof_lgb_te_seed{seed}.npy', oof_preds)

    mean_oof = np.mean(all_oof, axis=0)
    mean_test = np.mean(all_test, axis=0)

    mean_rmse = np.sqrt(mean_squared_error(y_train, mean_oof))
    print(f"\nLightGBM TE Mean CV RMSE: {mean_rmse:.4f}")

    np.save('oof_lgb_te_mean.npy', mean_oof)

    return mean_oof, mean_test, all_oof, all_test


def train_ridge_stacker(oof_preds_list, y_train, test_preds_list,
                        extra_features_train=None, extra_features_test=None):
    """Train Ridge stacker on OOF predictions."""
    print("\n" + "="*60)
    print("Training Ridge Stacker")
    print("="*60)

    # Stack OOF predictions as features
    X_stack = np.column_stack(oof_preds_list)
    X_test_stack = np.column_stack(test_preds_list)

    # Optionally add extra features (study_hours, attendance)
    if extra_features_train is not None:
        X_stack = np.column_stack([X_stack, extra_features_train])
        X_test_stack = np.column_stack([X_test_stack, extra_features_test])

    print(f"Stacker features: {X_stack.shape[1]}")

    # Simple Ridge regression
    stacker = Ridge(alpha=1.0)
    stacker.fit(X_stack, y_train)

    stacked_oof = stacker.predict(X_stack)
    stacked_test = stacker.predict(X_test_stack)

    stacked_rmse = np.sqrt(mean_squared_error(y_train, stacked_oof))
    print(f"Stacked CV RMSE: {stacked_rmse:.4f}")

    return stacked_oof, stacked_test, stacker


def linear_calibration(oof_preds, y_train, test_preds):
    """Fit linear calibration y = a*pred + b on OOF, apply to test."""
    print("\n" + "="*60)
    print("Linear Calibration")
    print("="*60)

    # Fit a, b
    X = np.column_stack([oof_preds, np.ones(len(oof_preds))])
    coeffs = np.linalg.lstsq(X, y_train, rcond=None)[0]
    a, b = coeffs[0], coeffs[1]

    print(f"Calibration: y = {a:.4f} * pred + {b:.4f}")

    calibrated_oof = a * oof_preds + b
    calibrated_test = a * test_preds + b

    before_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    after_rmse = np.sqrt(mean_squared_error(y_train, calibrated_oof))

    print(f"Before calibration RMSE: {before_rmse:.4f}")
    print(f"After calibration RMSE:  {after_rmse:.4f}")

    return calibrated_oof, calibrated_test


def clip_predictions(preds, clip_min=19, clip_max=100):
    """Clip predictions to valid range."""
    return np.clip(preds, clip_min, clip_max)


def create_submission(test_ids, predictions, filename):
    pd.DataFrame({'id': test_ids, 'exam_score': predictions}).to_csv(filename, index=False)
    print(f"Saved {filename}")


def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")

    # Print target range and verify clip bounds
    target_min = train['exam_score'].min()
    target_max = train['exam_score'].max()
    print(f"\nTarget range: [{target_min}, {target_max}]")
    print(f"Clip range:   [{CLIP_MIN}, {CLIP_MAX}]")
    assert CLIP_MIN <= target_min, f"CLIP_MIN ({CLIP_MIN}) > target min ({target_min})"
    assert CLIP_MAX >= target_max, f"CLIP_MAX ({CLIP_MAX}) < target max ({target_max})"
    print("✓ Clip range verified")

    # Get consistent fold indices
    fold_indices = get_fold_indices(len(train), N_FOLDS, random_state=42)

    # ========== CATBOOST ==========
    X_train_cat, y_train, X_test_cat, test_ids, cat_cols, num_cols = preprocess_catboost(train, test)

    cat_oof, cat_test, cat_all_oof, cat_all_test = train_catboost(
        X_train_cat, y_train, X_test_cat, cat_cols, fold_indices, CATBOOST_SEEDS
    )

    # ========== LIGHTGBM WITH TARGET ENCODING ==========
    X_train_lgb, X_test_lgb = preprocess_lgb_target_encoded(
        train, test, y_train, fold_indices, m=TARGET_SMOOTHING_VALUES[0]
    )

    lgb_oof, lgb_test, lgb_all_oof, lgb_all_test = train_lgb_target_encoded(
        X_train_lgb, y_train, X_test_lgb, fold_indices, LGB_SEEDS
    )

    # ========== SIMPLE AVERAGE ENSEMBLE ==========
    print("\n" + "="*60)
    print("Simple Average Ensemble")
    print("="*60)

    avg_oof = (cat_oof + lgb_oof) / 2
    avg_test = (cat_test + lgb_test) / 2
    avg_rmse = np.sqrt(mean_squared_error(y_train, avg_oof))
    print(f"Simple Average CV RMSE: {avg_rmse:.4f}")

    # ========== RIDGE STACKER ==========
    # Option 1: Just model predictions
    stacked_oof, stacked_test, stacker = train_ridge_stacker(
        [cat_oof, lgb_oof], y_train, [cat_test, lgb_test]
    )

    # Option 2: Add study_hours and attendance
    extra_train = train[['study_hours', 'class_attendance']].values
    extra_test = test[['study_hours', 'class_attendance']].values

    stacked_oof_v2, stacked_test_v2, stacker_v2 = train_ridge_stacker(
        [cat_oof, lgb_oof], y_train, [cat_test, lgb_test],
        extra_train, extra_test
    )

    # Pick best stacker
    if np.sqrt(mean_squared_error(y_train, stacked_oof_v2)) < np.sqrt(mean_squared_error(y_train, stacked_oof)):
        print("Using stacker with extra features")
        stacked_oof, stacked_test = stacked_oof_v2, stacked_test_v2

    # ========== LINEAR CALIBRATION ==========
    cal_oof, cal_test = linear_calibration(stacked_oof, y_train, stacked_test)

    # ========== CLIPPING ==========
    final_oof = clip_predictions(cal_oof, CLIP_MIN, CLIP_MAX)
    final_test = clip_predictions(cal_test, CLIP_MIN, CLIP_MAX)

    final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))
    print(f"\nFinal (calibrated + clipped) CV RMSE: {final_rmse:.4f}")

    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    cat_rmse = np.sqrt(mean_squared_error(y_train, cat_oof))
    lgb_rmse = np.sqrt(mean_squared_error(y_train, lgb_oof))

    print(f"CatBoost Mean ({len(CATBOOST_SEEDS)} seeds): {cat_rmse:.4f}")
    print(f"LightGBM TE Mean ({len(LGB_SEEDS)} seeds):   {lgb_rmse:.4f}")
    print(f"Simple Average:                   {avg_rmse:.4f}")
    print(f"Ridge Stacker:                    {np.sqrt(mean_squared_error(y_train, stacked_oof)):.4f}")
    print(f"Final (cal + clip):               {final_rmse:.4f}")
    print(f"\nPrevious best (Exp 07):           8.7395")
    print(f"Improvement:                      {8.7395 - final_rmse:+.4f}")

    # ========== SAVE ==========
    np.save('oof_final.npy', final_oof)
    np.save('y_train.npy', y_train)

    # Submissions
    create_submission(test_ids, cat_test, 'submission_catboost.csv')
    create_submission(test_ids, lgb_test, 'submission_lgb_te.csv')
    create_submission(test_ids, avg_test, 'submission_average.csv')
    create_submission(test_ids, clip_predictions(stacked_test, CLIP_MIN, CLIP_MAX), 'submission_stacked.csv')
    create_submission(test_ids, final_test, 'submission_final.csv')


if __name__ == '__main__':
    main()
