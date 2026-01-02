"""
Model averaging script - blend OOF predictions from all experiments.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Load all OOF predictions
y_train = np.load('experiments/01_baseline_lgb_xgb_catboost_ensemble/y_train.npy')

oof_models = {
    'lgb': np.load('experiments/01_baseline_lgb_xgb_catboost_ensemble/oof_lgb.npy'),
    'xgb': np.load('experiments/01_baseline_lgb_xgb_catboost_ensemble/oof_xgb.npy'),
    'ridge': np.load('experiments/02_polynomial_elasticnet/oof_ridge_deg3.npy'),
    'mlp': np.load('experiments/04_neural_network/oof_mlp.npy'),
    'lgb_emb': np.load('experiments/05_embedding_lgb/oof_lgb_emb.npy'),
}

# Print individual scores
print("Individual Model Scores:")
print("=" * 40)
for name, oof in oof_models.items():
    rmse = np.sqrt(mean_squared_error(y_train, oof))
    print(f"{name:12s}: {rmse:.4f}")

# Optimize weights
def rmse_loss(weights):
    weights = np.array(weights) / np.sum(weights)
    blended = sum(w * oof for w, oof in zip(weights, oof_models.values()))
    return np.sqrt(mean_squared_error(y_train, blended))

n = len(oof_models)
result = minimize(rmse_loss, [1/n]*n, bounds=[(0,1)]*n, method='SLSQP')
weights = np.array(result.x) / np.sum(result.x)

print("\n" + "=" * 40)
print("Optimized Weights:")
for name, w in zip(oof_models.keys(), weights):
    print(f"{name:12s}: {w:.3f}")

# Final blended score
blended_oof = sum(w * oof for w, oof in zip(weights, oof_models.values()))
print(f"\nBlended CV RMSE: {np.sqrt(mean_squared_error(y_train, blended_oof)):.4f}")
