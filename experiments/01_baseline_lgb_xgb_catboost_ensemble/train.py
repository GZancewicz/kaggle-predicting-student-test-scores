"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Fast version with hand-tuned parameters (no Optuna)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Try importing CatBoost
CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not installed. Run: pip install catboost")


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def create_features(df):
    """Feature engineering."""
    df = df.copy()

    # Mappings
    sleep_quality_map = {'poor': 1, 'average': 2, 'good': 3}
    internet_map = {'no': 0, 'yes': 1}
    facility_map = {'low': 1, 'medium': 2, 'high': 3}
    difficulty_map = {'easy': 1, 'moderate': 2, 'hard': 3}
    method_map = {'self-study': 1, 'online videos': 2, 'group study': 3, 'coaching': 4, 'mixed': 5}

    # Numeric encodings
    df['sleep_quality_num'] = df['sleep_quality'].map(sleep_quality_map)
    df['internet_num'] = df['internet_access'].map(internet_map)
    df['facility_num'] = df['facility_rating'].map(facility_map)
    df['difficulty_num'] = df['exam_difficulty'].map(difficulty_map)
    df['method_num'] = df['study_method'].map(method_map)

    # Interactions
    df['study_efficiency'] = df['study_hours'] * df['class_attendance'] / 100
    df['sleep_score'] = df['sleep_hours'] * df['sleep_quality_num']
    df['resource_score'] = df['internet_num'] * df['facility_num']
    df['adjusted_study'] = df['study_hours'] / df['difficulty_num']

    # Binned features
    df['attendance_bin'] = pd.cut(df['class_attendance'], bins=[0, 60, 80, 90, 100], labels=[1, 2, 3, 4]).astype(float)
    df['study_hours_bin'] = pd.cut(df['study_hours'], bins=[0, 3, 5, 7, 10], labels=[1, 2, 3, 4]).astype(float)

    # Combined scores
    df['total_input'] = df['study_efficiency'] * df['sleep_score']
    df['preparation_score'] = df['study_efficiency'] * 0.4 + df['resource_score'] * 0.3 + df['sleep_score'] * 0.3

    return df


def preprocess(train, test):
    train = create_features(train)
    test = create_features(test)

    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']

    # Label encode
    for col in cat_cols:
        le = LabelEncoder()
        train[col + '_enc'] = le.fit_transform(train[col].astype(str))
        test[col + '_enc'] = le.transform(test[col].astype(str))

    feature_cols = [
        'age', 'study_hours', 'class_attendance', 'sleep_hours',
        'sleep_quality_num', 'internet_num', 'facility_num', 'difficulty_num', 'method_num',
        'study_efficiency', 'sleep_score', 'resource_score', 'adjusted_study',
        'attendance_bin', 'study_hours_bin', 'total_input', 'preparation_score'
    ] + [col + '_enc' for col in cat_cols]

    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train['exam_score'].values
    X_test = test[feature_cols].values.astype(np.float32)
    test_ids = test['id'].values

    # For CatBoost
    X_train_cat = train[cat_cols]
    X_test_cat = test[cat_cols]

    return X_train, y_train, X_test, test_ids, feature_cols, X_train_cat, X_test_cat, cat_cols


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


def train_catboost(X_train_cat, y_train, X_test_cat, cat_cols, n_folds=5):
    """CatBoost with native categorical handling."""
    if not CATBOOST_AVAILABLE:
        print("CatBoost not available")
        return None, None

    print("\n" + "="*50)
    print("Training CatBoost")
    print("="*50)

    X_train_np = X_train_cat.values
    X_test_np = X_test_cat.values
    cat_features = list(range(len(cat_cols)))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train_np))
    test_preds = np.zeros(len(X_test_np))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
        X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            early_stopping_rounds=100,
            verbose=200,
            cat_features=cat_features
        )

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test_np) / n_folds
        print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, oof_preds[val_idx])):.4f}")

    print(f"\nCatBoost CV RMSE: {np.sqrt(mean_squared_error(y_train, oof_preds)):.4f}")
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

    print("\nPreprocessing...")
    X_train, y_train, X_test, test_ids, features, X_train_cat, X_test_cat, cat_cols = preprocess(train, test)
    print(f"Features: {len(features)}")

    # Train all models
    lgb_oof, lgb_test = train_lightgbm(X_train, y_train, X_test)
    xgb_oof, xgb_test = train_xgboost(X_train, y_train, X_test)
    cat_oof, cat_test = train_catboost(X_train_cat, y_train, X_test_cat, cat_cols)

    # Ensemble
    print("\n" + "="*50)
    print("Creating Ensemble")
    print("="*50)

    oof_list = [lgb_oof, xgb_oof]
    test_list = [lgb_test, xgb_test]
    names = ['LGB', 'XGB']

    if cat_oof is not None:
        oof_list.append(cat_oof)
        test_list.append(cat_test)
        names.append('CAT')

    weights = optimize_weights(oof_list, y_train)
    print(f"Optimal weights: {dict(zip(names, weights.round(3)))}")

    final_oof = sum(w * p for w, p in zip(weights, oof_list))
    final_test = sum(w * p for w, p in zip(weights, test_list))

    print(f"\nFinal Ensemble CV RMSE: {np.sqrt(mean_squared_error(y_train, final_oof)):.4f}")

    # Save OOF predictions for model averaging
    np.save('oof_lgb.npy', lgb_oof)
    np.save('oof_xgb.npy', xgb_oof)
    np.save('oof_ensemble.npy', final_oof)
    np.save('y_train.npy', y_train)
    if cat_oof is not None:
        np.save('oof_cat.npy', cat_oof)
    print("Saved OOF predictions (.npy files)")

    # Save submissions
    create_submission(test_ids, lgb_test, 'submission_lgb.csv')
    create_submission(test_ids, xgb_test, 'submission_xgb.csv')
    if cat_test is not None:
        create_submission(test_ids, cat_test, 'submission_cat.csv')
    create_submission(test_ids, final_test, 'submission_ensemble.csv')


if __name__ == '__main__':
    main()
