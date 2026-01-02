"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 12: Multi-Seed Stacking

Goal: Reduce variance via multi-seed averaging, then re-stack.
No feature changes, no fold changes, no hyperparameter tuning.

Components:
1. CatBoost with native categoricals + ordered boosting (3 seeds)
2. LightGBM GOSS with OOF target encoding (3 seeds)
3. Ridge stacker on seed-averaged OOF predictions
4. Linear calibration + clipping
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration - SAME AS EXP 11
N_FOLDS = 5
SEEDS = [42, 202, 999]  # 3 seeds for variance reduction
TARGET_SMOOTHING = 10
CLIP_MIN, CLIP_MAX = 19, 100


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def get_fold_indices(n_samples, n_folds=5, random_state=42):
    """Get consistent fold indices - SAME AS EXP 11."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


def target_encode_oof(train_col, y, fold_indices, m=10):
    """OOF-safe target encoding with smoothing, no noise."""
    global_mean = y.mean()
    encoded = np.zeros(len(train_col))

    for train_idx, val_idx in fold_indices:
        df_fold = pd.DataFrame({'cat': train_col.iloc[train_idx], 'y': y[train_idx]})
        stats = df_fold.groupby('cat')['y'].agg(['sum', 'count'])
        stats['enc'] = (stats['sum'] + m * global_mean) / (stats['count'] + m)
        encoded[val_idx] = train_col.iloc[val_idx].map(stats['enc']).fillna(global_mean).values

    return encoded


def target_encode_test(train_col, y, test_col, m=10):
    """Target encoding for test set using full training data."""
    global_mean = y.mean()
    df = pd.DataFrame({'cat': train_col, 'y': y})
    stats = df.groupby('cat')['y'].agg(['sum', 'count'])
    stats['enc'] = (stats['sum'] + m * global_mean) / (stats['count'] + m)
    return test_col.map(stats['enc']).fillna(global_mean).values


def frequency_encode(train_col, test_col):
    """Frequency encoding."""
    freq = train_col.value_counts().to_dict()
    train_enc = train_col.map(freq).values
    test_enc = test_col.map(freq).fillna(0).values
    return train_enc, test_enc


def preprocess_catboost(train, test):
    """Prepare data for CatBoost with native categoricals."""
    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']
    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

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

    for col in cat_cols:
        X_train[f'{col}_te'] = target_encode_oof(train[col], y_train, fold_indices, m=m)
        X_test[f'{col}_te'] = target_encode_test(train[col], y_train, test[col], m=m)

        train_freq, test_freq = frequency_encode(train[col], test[col])
        X_train[f'{col}_freq'] = train_freq
        X_test[f'{col}_freq'] = test_freq

    return X_train.values.astype(np.float32), X_test.values.astype(np.float32)


def train_catboost_multiseed(X_train, y_train, X_test, cat_cols, fold_indices, seeds):
    """Train CatBoost with multiple seeds, return per-seed and averaged predictions."""
    print("\n" + "="*70)
    print(f"CATBOOST - {len(seeds)} seeds x {len(fold_indices)} folds")
    print("="*70)

    cat_features = [X_train.columns.get_loc(c) for c in cat_cols]

    all_seed_oof = []
    all_seed_test = []
    seed_rmses = []

    start_time = time.time()

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))
        fold_rmses = []

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
                boosting_type='Ordered',
                od_type='Iter',
                od_wait=100,
                verbose=200,
                cat_features=cat_features
            )

            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / len(fold_indices)

            fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
            fold_rmses.append(fold_rmse)
            print(f"  Fold {fold+1}: {fold_rmse:.4f}")

        seed_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
        seed_rmses.append(seed_rmse)
        print(f"  Seed {seed} CV: {seed_rmse:.4f} (std: {np.std(fold_rmses):.4f})")

        all_seed_oof.append(oof_preds)
        all_seed_test.append(test_preds)

    elapsed = time.time() - start_time

    # Average across seeds
    mean_oof = np.mean(all_seed_oof, axis=0)
    mean_test = np.mean(all_seed_test, axis=0)
    mean_rmse = np.sqrt(mean_squared_error(y_train, mean_oof))

    print(f"\n>>> CatBoost Summary:")
    print(f"    Per-seed RMSEs: {[f'{r:.4f}' for r in seed_rmses]}")
    print(f"    Seed-averaged CV: {mean_rmse:.4f}")
    print(f"    Seed std: {np.std(seed_rmses):.4f}")
    print(f"    Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return mean_oof, mean_test, all_seed_oof, all_seed_test, seed_rmses, elapsed


def train_lgb_multiseed(X_train, y_train, X_test, fold_indices, seeds):
    """Train LightGBM GOSS with multiple seeds."""
    print("\n" + "="*70)
    print(f"LIGHTGBM GOSS + TARGET ENCODING - {len(seeds)} seeds x {len(fold_indices)} folds")
    print("="*70)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'goss',
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

    all_seed_oof = []
    all_seed_test = []
    seed_rmses = []

    start_time = time.time()

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        params['seed'] = seed

        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                params, train_data,
                num_boost_round=2000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )

            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / len(fold_indices)

            fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
            fold_rmses.append(fold_rmse)
            print(f"  Fold {fold+1}: {fold_rmse:.4f}")

        seed_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
        seed_rmses.append(seed_rmse)
        print(f"  Seed {seed} CV: {seed_rmse:.4f} (std: {np.std(fold_rmses):.4f})")

        all_seed_oof.append(oof_preds)
        all_seed_test.append(test_preds)

    elapsed = time.time() - start_time

    # Average across seeds
    mean_oof = np.mean(all_seed_oof, axis=0)
    mean_test = np.mean(all_seed_test, axis=0)
    mean_rmse = np.sqrt(mean_squared_error(y_train, mean_oof))

    print(f"\n>>> LightGBM Summary:")
    print(f"    Per-seed RMSEs: {[f'{r:.4f}' for r in seed_rmses]}")
    print(f"    Seed-averaged CV: {mean_rmse:.4f}")
    print(f"    Seed std: {np.std(seed_rmses):.4f}")
    print(f"    Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return mean_oof, mean_test, all_seed_oof, all_seed_test, seed_rmses, elapsed


def train_ridge_stacker(oof_list, y_train, test_list, extra_train=None, extra_test=None):
    """Train Ridge stacker on seed-averaged OOF predictions."""
    print("\n" + "="*70)
    print("RIDGE STACKER")
    print("="*70)

    X_stack = np.column_stack(oof_list)
    X_test_stack = np.column_stack(test_list)

    if extra_train is not None:
        X_stack = np.column_stack([X_stack, extra_train])
        X_test_stack = np.column_stack([X_test_stack, extra_test])
        print(f"Features: {X_stack.shape[1]} (2 model preds + {extra_train.shape[1]} extra)")
    else:
        print(f"Features: {X_stack.shape[1]} (model predictions only)")

    stacker = Ridge(alpha=1.0)
    stacker.fit(X_stack, y_train)

    stacked_oof = stacker.predict(X_stack)
    stacked_test = stacker.predict(X_test_stack)

    stacked_rmse = np.sqrt(mean_squared_error(y_train, stacked_oof))
    print(f"Stacked CV RMSE: {stacked_rmse:.4f}")

    return stacked_oof, stacked_test, stacker


def linear_calibration(oof_preds, y_train, test_preds):
    """Fit linear calibration on OOF, apply to test."""
    print("\n" + "="*70)
    print("LINEAR CALIBRATION")
    print("="*70)

    X = np.column_stack([oof_preds, np.ones(len(oof_preds))])
    coeffs = np.linalg.lstsq(X, y_train, rcond=None)[0]
    a, b = coeffs[0], coeffs[1]

    print(f"Calibration: y = {a:.4f} * pred + {b:.4f}")

    calibrated_oof = a * oof_preds + b
    calibrated_test = a * test_preds + b

    before_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    after_rmse = np.sqrt(mean_squared_error(y_train, calibrated_oof))

    print(f"Before calibration: {before_rmse:.4f}")
    print(f"After calibration:  {after_rmse:.4f}")

    return calibrated_oof, calibrated_test


def create_submission(test_ids, predictions, filename):
    pd.DataFrame({'id': test_ids, 'exam_score': predictions}).to_csv(filename, index=False)
    print(f"Saved {filename}")


def main():
    total_start = time.time()

    print("="*70)
    print("EXPERIMENT 12: MULTI-SEED STACKING")
    print("Goal: Variance reduction via 3-seed averaging")
    print("="*70)

    print("\nLoading data...")
    train, test = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")

    # Verify target range
    target_min, target_max = train['exam_score'].min(), train['exam_score'].max()
    print(f"Target range: [{target_min}, {target_max}]")
    print(f"Clip range:   [{CLIP_MIN}, {CLIP_MAX}]")

    # Get consistent fold indices - SAME AS EXP 11
    fold_indices = get_fold_indices(len(train), N_FOLDS, random_state=42)

    # ========== CATBOOST ==========
    X_train_cat, y_train, X_test_cat, test_ids, cat_cols, num_cols = preprocess_catboost(train, test)

    cat_oof, cat_test, cat_all_oof, cat_all_test, cat_seed_rmses, cat_time = train_catboost_multiseed(
        X_train_cat, y_train, X_test_cat, cat_cols, fold_indices, SEEDS
    )

    # ========== LIGHTGBM WITH TARGET ENCODING ==========
    X_train_lgb, X_test_lgb = preprocess_lgb_target_encoded(
        train, test, y_train, fold_indices, m=TARGET_SMOOTHING
    )

    lgb_oof, lgb_test, lgb_all_oof, lgb_all_test, lgb_seed_rmses, lgb_time = train_lgb_multiseed(
        X_train_lgb, y_train, X_test_lgb, fold_indices, SEEDS
    )

    # ========== SIMPLE AVERAGE ==========
    print("\n" + "="*70)
    print("SIMPLE AVERAGE ENSEMBLE")
    print("="*70)
    avg_oof = (cat_oof + lgb_oof) / 2
    avg_test = (cat_test + lgb_test) / 2
    avg_rmse = np.sqrt(mean_squared_error(y_train, avg_oof))
    print(f"Simple Average CV: {avg_rmse:.4f}")

    # ========== RIDGE STACKER ==========
    # Version 1: Just model predictions
    stacked_oof_v1, stacked_test_v1, _ = train_ridge_stacker(
        [cat_oof, lgb_oof], y_train, [cat_test, lgb_test]
    )

    # Version 2: With extra features
    extra_train = train[['study_hours', 'class_attendance']].values
    extra_test = test[['study_hours', 'class_attendance']].values

    stacked_oof_v2, stacked_test_v2, _ = train_ridge_stacker(
        [cat_oof, lgb_oof], y_train, [cat_test, lgb_test],
        extra_train, extra_test
    )

    # Pick best stacker
    rmse_v1 = np.sqrt(mean_squared_error(y_train, stacked_oof_v1))
    rmse_v2 = np.sqrt(mean_squared_error(y_train, stacked_oof_v2))

    if rmse_v2 < rmse_v1:
        print(f"\nUsing stacker V2 (with extra features): {rmse_v2:.4f}")
        stacked_oof, stacked_test = stacked_oof_v2, stacked_test_v2
    else:
        print(f"\nUsing stacker V1 (predictions only): {rmse_v1:.4f}")
        stacked_oof, stacked_test = stacked_oof_v1, stacked_test_v1

    # ========== LINEAR CALIBRATION ==========
    cal_oof, cal_test = linear_calibration(stacked_oof, y_train, stacked_test)

    # ========== CLIPPING ==========
    final_oof = np.clip(cal_oof, CLIP_MIN, CLIP_MAX)
    final_test = np.clip(cal_test, CLIP_MIN, CLIP_MAX)
    final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))

    total_time = time.time() - total_start

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    cat_rmse = np.sqrt(mean_squared_error(y_train, cat_oof))
    lgb_rmse = np.sqrt(mean_squared_error(y_train, lgb_oof))
    stacked_rmse = np.sqrt(mean_squared_error(y_train, stacked_oof))

    print(f"\nCatBoost ({len(SEEDS)} seeds):")
    print(f"  Per-seed: {[f'{r:.4f}' for r in cat_seed_rmses]}")
    print(f"  Averaged: {cat_rmse:.4f}")
    print(f"  Seed std: {np.std(cat_seed_rmses):.4f}")
    print(f"  Runtime:  {cat_time:.1f}s")

    print(f"\nLightGBM TE ({len(SEEDS)} seeds):")
    print(f"  Per-seed: {[f'{r:.4f}' for r in lgb_seed_rmses]}")
    print(f"  Averaged: {lgb_rmse:.4f}")
    print(f"  Seed std: {np.std(lgb_seed_rmses):.4f}")
    print(f"  Runtime:  {lgb_time:.1f}s")

    print(f"\nEnsembles:")
    print(f"  Simple Average:     {avg_rmse:.4f}")
    print(f"  Ridge Stacker:      {stacked_rmse:.4f}")
    print(f"  After calibration:  {np.sqrt(mean_squared_error(y_train, cal_oof)):.4f}")
    print(f"  Final (clipped):    {final_rmse:.4f}")

    print(f"\nComparison to previous:")
    print(f"  Exp 11 (1 seed):  8.7604")
    print(f"  Exp 12 (3 seeds): {final_rmse:.4f}")
    print(f"  Improvement:      {8.7604 - final_rmse:+.4f}")

    print(f"\nExp 07 (best):      8.7395")
    print(f"Exp 12 vs Exp 07:   {8.7395 - final_rmse:+.4f}")

    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Decision rule
    improvement = 8.7604 - final_rmse
    print(f"\n>>> DECISION:")
    if improvement >= 0.03:
        print(f"    Improvement >= 0.03 ({improvement:.4f}) -> Scale to 5-10 seeds")
    elif improvement >= 0.02:
        print(f"    Improvement 0.02-0.03 ({improvement:.4f}) -> Consider more seeds")
    else:
        print(f"    Improvement < 0.02 ({improvement:.4f}) -> Stop scaling, revisit strategy")

    # ========== SAVE OUTPUTS ==========
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    np.save('oof_catboost_seedavg.npy', cat_oof)
    np.save('oof_lgb_te_seedavg.npy', lgb_oof)
    np.save('oof_stacked_seedavg.npy', stacked_oof)
    np.save('oof_final.npy', final_oof)
    np.save('y_train.npy', y_train)

    create_submission(test_ids, np.clip(cat_test, CLIP_MIN, CLIP_MAX), 'submission_catboost_seedavg.csv')
    create_submission(test_ids, np.clip(lgb_test, CLIP_MIN, CLIP_MAX), 'submission_lgb_te_seedavg.csv')
    create_submission(test_ids, np.clip(stacked_test, CLIP_MIN, CLIP_MAX), 'submission_stacked_seedavg.csv')
    create_submission(test_ids, final_test, 'submission_final.csv')

    print("\nDone!")


if __name__ == '__main__':
    main()
