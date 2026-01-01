# Experiment 02: Polynomial Features + ElasticNet/Ridge

## Summary
| Model | CV RMSE |
|-------|---------|
| ElasticNet (degree 2) | 9.1617 |
| Ridge (degree 2) | 9.0515 |
| **Ridge (degree 3)** | **8.9818** |

**Best: Ridge degree 3 (8.98)** - Still worse than tree ensemble (8.74)

## Description
Power series approximation to f(x₁, x₂, ...) using polynomial feature expansion:
- 1st order: x₁, x₂, ...
- 2nd order: x₁², x₂², x₁x₂, ...
- 3rd order: x₁³, x₁²x₂, x₁x₂x₃, ...

## Feature Expansion
- Original: 11 features
- Degree 2: 77 features (11 linear + 11 quadratic + 55 interactions)
- Degree 3: ~364 features

## Models
- **ElasticNet**: L1 + L2 regularization (feature selection + shrinkage)
- **Ridge**: L2 regularization only (shrinkage, keeps all features)

## Results

### Per-Fold RMSE

| Fold | ElasticNet (deg 2) | Ridge (deg 2) | Ridge (deg 3) |
|------|-------------------|---------------|---------------|
| 1    | 9.1485 | 9.0357 | 8.9702 |
| 2    | 9.1553 | 9.0421 | 8.9752 |
| 3    | 9.1567 | 9.0514 | 8.9788 |
| 4    | 9.1633 | 9.0526 | 8.9787 |
| 5    | 9.1848 | 9.0755 | 9.0060 |

## Training
- StandardScaler normalization (required for linear models)
- 5-fold cross-validation
- Fixed hyperparameters (alpha=0.1 for ElasticNet, alpha=1.0 for Ridge)

## Observations
- **Ridge > ElasticNet**: With many correlated polynomial features, Ridge (L2) outperforms Lasso-style sparsity
- **Higher degree helps**: Degree 3 beats degree 2 by ~0.07 RMSE
- **Still behind trees**: Best linear model (8.98) is 0.24 RMSE worse than LGB/XGB ensemble (8.74)
- **Diminishing returns**: Could try degree 4, but likely won't close the gap to trees

## Files
- `train.py` - Main training script
- `submission_ridge_deg3.csv` - Best submission

## Run
```bash
cd experiments/02_polynomial_elasticnet
python train.py
```
