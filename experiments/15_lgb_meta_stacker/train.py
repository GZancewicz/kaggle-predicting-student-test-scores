"""
Experiment 15: LightGBM Meta-Stacker

Test whether a nonlinear meta-model (small LightGBM) can improve over Ridge stacking.

Inputs:
- catboost_oof_seedavg
- lgb_te_oof_seedavg
- Optional: abs(cat - lgb), cat - lgb

Stacker: Very small LightGBM with strong regularization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_FOLDS = 5
SEEDS = [42, 202, 999]  # 3 seeds for base models
CLIP_MIN, CLIP_MAX = 19, 100

# LightGBM meta-stacker params (very small, heavily regularized)
LGB_STACKER_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 7,
    'max_depth': 3,
    'min_data_in_leaf': 2000,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'lambda_l2': 1.0,
    'verbosity': -1,
    'seed': 42
}


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def get_fold_indices(n_samples, n_folds=5, random_state=42):
    """Get consistent fold indices for all models."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


def target_encode_oof(train_col, y, fold_indices, m=10):
    """OOF-safe target encoding with smoothing."""
    global_mean = y.mean()
    encoded = np.zeros(len(train_col))

    for train_idx, val_idx in fold_indices:
        train_df = pd.DataFrame({'col': train_col.iloc[train_idx], 'y': y.iloc[train_idx]})
        stats = train_df.groupby('col')['y'].agg(['sum', 'count'])

        for val_i in val_idx:
            cat_val = train_col.iloc[val_i]
            if cat_val in stats.index:
                sum_y = stats.loc[cat_val, 'sum']
                count = stats.loc[cat_val, 'count']
                encoded[val_i] = (sum_y + m * global_mean) / (count + m)
            else:
                encoded[val_i] = global_mean

    return encoded


def target_encode_test(train_col, y, test_col, m=10):
    """Target encoding for test set using full train."""
    global_mean = y.mean()
    train_df = pd.DataFrame({'col': train_col, 'y': y})
    stats = train_df.groupby('col')['y'].agg(['sum', 'count'])

    encoded = np.zeros(len(test_col))
    for i, cat_val in enumerate(test_col):
        if cat_val in stats.index:
            sum_y = stats.loc[cat_val, 'sum']
            count = stats.loc[cat_val, 'count']
            encoded[i] = (sum_y + m * global_mean) / (count + m)
        else:
            encoded[i] = global_mean

    return encoded


def preprocess_catboost(train, test):
    """Prepare data for CatBoost (native categoricals)."""
    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']
    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    X_train = train[cat_cols + num_cols].copy()
    y_train = train['exam_score'].copy()
    X_test = test[cat_cols + num_cols].copy()
    test_ids = test['id'].copy()

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    return X_train, y_train, X_test, test_ids, cat_cols, num_cols


def preprocess_lgb_te(train, test, y, fold_indices, m=10):
    """Prepare data for LightGBM with target encoding."""
    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']
    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    X_train = train[num_cols].copy()
    X_test = test[num_cols].copy()

    for col in cat_cols:
        X_train[f'{col}_te'] = target_encode_oof(train[col], y, fold_indices, m=m)
        X_test[f'{col}_te'] = target_encode_test(train[col], y, test[col], m=m)

    return X_train, X_test


def train_catboost(X_train, y_train, X_test, cat_cols, fold_indices, seeds):
    """Train CatBoost with multiple seeds, return seed-averaged OOF and test predictions."""
    n_samples = len(X_train)
    cat_features = [X_train.columns.get_loc(c) for c in cat_cols]

    all_oof = []
    all_test = []

    for seed in seeds:
        print(f"\n  CatBoost seed {seed}...")
        oof_preds = np.zeros(n_samples)
        test_preds = np.zeros(len(X_test))

        for fold_num, (train_idx, val_idx) in enumerate(fold_indices):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

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

        rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
        print(f"    Seed {seed} CV RMSE: {rmse:.4f}")

        all_oof.append(oof_preds)
        all_test.append(test_preds)

    # Seed average
    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)

    return oof_avg, test_avg


def train_lgb_te(X_train, y_train, X_test, fold_indices, seeds):
    """Train LightGBM with target encoding, multiple seeds."""
    n_samples = len(X_train)

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'goss',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': -1,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'top_rate': 0.2,
        'other_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbosity': -1
    }

    all_oof = []
    all_test = []

    for seed in seeds:
        print(f"\n  LightGBM seed {seed}...")
        lgb_params['seed'] = seed

        oof_preds = np.zeros(n_samples)
        test_preds = np.zeros(len(X_test))

        for fold_num, (train_idx, val_idx) in enumerate(fold_indices):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=2000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
            )

            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / len(fold_indices)

        rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
        print(f"    Seed {seed} CV RMSE: {rmse:.4f}")

        all_oof.append(oof_preds)
        all_test.append(test_preds)

    # Seed average
    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)

    return oof_avg, test_avg


def train_ridge_stacker(cat_oof, lgb_oof, y_train, cat_test, lgb_test, fold_indices):
    """Train Ridge stacker (baseline)."""
    # Build stacking features
    X_stack_train = np.column_stack([cat_oof, lgb_oof])
    X_stack_test = np.column_stack([cat_test, lgb_test])

    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(cat_test))

    for train_idx, val_idx in fold_indices:
        X_tr = X_stack_train[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_stack_train[val_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_stack_test) / len(fold_indices)

    return oof_preds, test_preds


def train_lgb_stacker(cat_oof, lgb_oof, y_train, cat_test, lgb_test, fold_indices, add_diff_features=True):
    """Train LightGBM meta-stacker."""
    # Build stacking features
    if add_diff_features:
        X_stack_train = np.column_stack([
            cat_oof,
            lgb_oof,
            cat_oof - lgb_oof,
            np.abs(cat_oof - lgb_oof)
        ])
        X_stack_test = np.column_stack([
            cat_test,
            lgb_test,
            cat_test - lgb_test,
            np.abs(cat_test - lgb_test)
        ])
        feature_names = ['cat_oof', 'lgb_oof', 'cat_minus_lgb', 'abs_diff']
    else:
        X_stack_train = np.column_stack([cat_oof, lgb_oof])
        X_stack_test = np.column_stack([cat_test, lgb_test])
        feature_names = ['cat_oof', 'lgb_oof']

    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(cat_test))
    fold_rmses = []

    for fold_num, (train_idx, val_idx) in enumerate(fold_indices):
        X_tr = X_stack_train[train_idx]
        y_tr = y_train.iloc[train_idx].values
        X_val = X_stack_train[val_idx]
        y_val = y_train.iloc[val_idx].values

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            LGB_STACKER_PARAMS,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_stack_test) / len(fold_indices)

        fold_rmse = np.sqrt(np.mean((oof_preds[val_idx] - y_val) ** 2))
        fold_rmses.append(fold_rmse)

    return oof_preds, test_preds, fold_rmses


def create_submission(test_ids, predictions, filename):
    """Create submission file."""
    sub = pd.DataFrame({'id': test_ids, 'exam_score': predictions})
    sub.to_csv(filename, index=False)
    print(f"Saved {filename}")


def main():
    print("=" * 60)
    print("Experiment 15: LightGBM Meta-Stacker")
    print("=" * 60)

    print("\nLoading data...")
    train, test = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")

    y_train = train['exam_score']

    # Get consistent fold indices
    fold_indices = get_fold_indices(len(train), N_FOLDS, random_state=42)

    # ========== STEP 1: Train Base Models ==========
    print("\n" + "=" * 60)
    print("Step 1: Training Base Models (3 seeds each)")
    print("=" * 60)

    # CatBoost
    print("\n>>> CatBoost (Ordered, native categoricals)")
    X_train_cat, y_train_cat, X_test_cat, test_ids, cat_cols, num_cols = preprocess_catboost(train, test)
    cat_oof, cat_test = train_catboost(X_train_cat, y_train_cat, X_test_cat, cat_cols, fold_indices, SEEDS)
    cat_rmse = np.sqrt(np.mean((cat_oof - y_train) ** 2))
    print(f"\nCatBoost seed-avg CV RMSE: {cat_rmse:.4f}")

    # LightGBM with target encoding
    print("\n>>> LightGBM (GOSS, target encoding)")
    X_train_lgb, X_test_lgb = preprocess_lgb_te(train, test, y_train, fold_indices, m=10)
    lgb_oof, lgb_test = train_lgb_te(X_train_lgb, y_train, X_test_lgb, fold_indices, SEEDS)
    lgb_rmse = np.sqrt(np.mean((lgb_oof - y_train) ** 2))
    print(f"\nLightGBM seed-avg CV RMSE: {lgb_rmse:.4f}")

    # ========== STEP 2: Stacking ==========
    print("\n" + "=" * 60)
    print("Step 2: Meta-Stacking")
    print("=" * 60)

    # Ridge stacker (baseline)
    print("\n>>> Ridge Stacker (baseline)")
    ridge_oof, ridge_test = train_ridge_stacker(cat_oof, lgb_oof, y_train, cat_test, lgb_test, fold_indices)
    ridge_rmse = np.sqrt(np.mean((ridge_oof - y_train) ** 2))
    print(f"Ridge Stacker CV RMSE: {ridge_rmse:.4f}")

    # LightGBM stacker (without diff features)
    print("\n>>> LightGBM Stacker (cat_oof, lgb_oof only)")
    lgb_stack_oof_nodiff, lgb_stack_test_nodiff, fold_rmses_nodiff = train_lgb_stacker(
        cat_oof, lgb_oof, y_train, cat_test, lgb_test, fold_indices, add_diff_features=False
    )
    lgb_stack_rmse_nodiff = np.sqrt(np.mean((lgb_stack_oof_nodiff - y_train) ** 2))

    # LightGBM stacker (with diff features)
    print("\n>>> LightGBM Stacker (with diff features)")
    lgb_stack_oof, lgb_stack_test, fold_rmses = train_lgb_stacker(
        cat_oof, lgb_oof, y_train, cat_test, lgb_test, fold_indices, add_diff_features=True
    )
    lgb_stack_rmse = np.sqrt(np.mean((lgb_stack_oof - y_train) ** 2))

    # ========== STEP 3: Results ==========
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print("\n| Model | CV RMSE |")
    print("|-------|---------|")
    print(f"| CatBoost (3 seeds) | {cat_rmse:.4f} |")
    print(f"| LightGBM TE (3 seeds) | {lgb_rmse:.4f} |")
    print(f"| Ridge Stacker | {ridge_rmse:.4f} |")
    print(f"| LGB Stacker (2 features) | {lgb_stack_rmse_nodiff:.4f} |")
    print(f"| LGB Stacker (4 features) | {lgb_stack_rmse:.4f} |")

    # Choose best LGB stacker
    if lgb_stack_rmse <= lgb_stack_rmse_nodiff:
        best_lgb_stack_oof = lgb_stack_oof
        best_lgb_stack_test = lgb_stack_test
        best_lgb_stack_rmse = lgb_stack_rmse
        best_fold_rmses = fold_rmses
        best_variant = "4 features"
    else:
        best_lgb_stack_oof = lgb_stack_oof_nodiff
        best_lgb_stack_test = lgb_stack_test_nodiff
        best_lgb_stack_rmse = lgb_stack_rmse_nodiff
        best_fold_rmses = fold_rmses_nodiff
        best_variant = "2 features"

    print(f"\nBest LGB stacker: {best_variant}")

    # Delta
    delta = ridge_rmse - best_lgb_stack_rmse
    print(f"\n>>> Delta (Ridge - LGB): {delta:.4f}")
    print(f">>> Fold RMSE std: {np.std(best_fold_rmses):.4f}")

    # Clip and final
    final_oof = np.clip(best_lgb_stack_oof, CLIP_MIN, CLIP_MAX)
    final_test = np.clip(best_lgb_stack_test, CLIP_MIN, CLIP_MAX)
    final_rmse = np.sqrt(np.mean((final_oof - y_train) ** 2))
    print(f"\n>>> Final (clipped) CV RMSE: {final_rmse:.4f}")

    # Decision rule
    print("\n" + "=" * 60)
    print("Decision Rule")
    print("=" * 60)
    if delta >= 0.01:
        print(f"✓ Improvement >= 0.01 ({delta:.4f})")
        print("→ Submit and stop")
    else:
        print(f"✗ Improvement < 0.01 ({delta:.4f})")
        print("→ Stop permanently")

    # ========== STEP 4: Submissions ==========
    print("\n" + "=" * 60)
    print("Creating Submissions")
    print("=" * 60)

    create_submission(test_ids, np.clip(cat_test, CLIP_MIN, CLIP_MAX), 'submission_catboost.csv')
    create_submission(test_ids, np.clip(lgb_test, CLIP_MIN, CLIP_MAX), 'submission_lgb_te.csv')
    create_submission(test_ids, np.clip(ridge_test, CLIP_MIN, CLIP_MAX), 'submission_ridge_stacker.csv')
    create_submission(test_ids, final_test, 'submission_lgb_stacker.csv')

    print("\nDone!")


if __name__ == '__main__':
    main()
