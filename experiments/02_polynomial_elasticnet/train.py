"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 02: 2nd Order Polynomial Features + ElasticNet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def preprocess(train, test):
    """Encode categoricals and prepare numeric features."""
    train = train.copy()
    test = test.copy()

    # Ordinal encodings (preserve order where meaningful)
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

    # Feature columns (all numeric now)
    feature_cols = ['age', 'gender', 'course', 'study_hours', 'class_attendance',
                    'internet_access', 'sleep_hours', 'sleep_quality',
                    'study_method', 'facility_rating', 'exam_difficulty']

    X_train = train[feature_cols].values
    y_train = train['exam_score'].values
    X_test = test[feature_cols].values
    test_ids = test['id'].values

    return X_train, y_train, X_test, test_ids, feature_cols


def train_polynomial_elasticnet(X_train, y_train, X_test, degree=2, n_folds=5):
    """Train ElasticNet with polynomial features."""
    print(f"\n{'='*50}")
    print(f"Polynomial Degree {degree} + ElasticNet")
    print('='*50)

    # Create polynomial features
    print(f"Original features: {X_train.shape[1]}")
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    print(f"Polynomial features: {X_train_poly.shape[1]}")

    # Scale features (important for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # ElasticNet with CV-tuned alpha would be better, but using reasonable defaults
        model = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,  # 0=Ridge, 1=Lasso, 0.5=balanced
            max_iter=1000,
            random_state=42
        )
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test_scaled) / n_folds

        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nElasticNet CV RMSE: {overall_rmse:.4f}")

    return oof_preds, test_preds, poly.get_feature_names_out()


def train_polynomial_ridge(X_train, y_train, X_test, degree=2, n_folds=5):
    """Train Ridge with polynomial features (often better than ElasticNet for dense features)."""
    print(f"\n{'='*50}")
    print(f"Polynomial Degree {degree} + Ridge")
    print('='*50)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test_scaled) / n_folds

        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nRidge CV RMSE: {overall_rmse:.4f}")

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
    print(f"Base features: {feature_cols}")

    # Train with degree 2
    elastic_oof, elastic_test, poly_features = train_polynomial_elasticnet(
        X_train, y_train, X_test, degree=2
    )

    ridge_oof, ridge_test = train_polynomial_ridge(
        X_train, y_train, X_test, degree=2
    )

    # Try degree 3 for comparison
    print("\n" + "="*50)
    print("Trying Degree 3...")
    print("="*50)
    ridge3_oof, ridge3_test = train_polynomial_ridge(
        X_train, y_train, X_test, degree=3
    )

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"ElasticNet (degree 2): {np.sqrt(mean_squared_error(y_train, elastic_oof)):.4f}")
    print(f"Ridge (degree 2):      {np.sqrt(mean_squared_error(y_train, ridge_oof)):.4f}")
    print(f"Ridge (degree 3):      {np.sqrt(mean_squared_error(y_train, ridge3_oof)):.4f}")

    # Save OOF predictions
    np.save('oof_ridge_deg3.npy', ridge3_oof)
    np.save('y_train.npy', y_train)
    print("Saved OOF predictions (.npy files)")

    # Save best submission
    create_submission(test_ids, ridge3_test, 'submission_ridge_deg3.csv')


if __name__ == '__main__':
    main()
