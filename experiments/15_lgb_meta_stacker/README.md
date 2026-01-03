# Experiment 15: LightGBM Meta-Stacker

Test whether a nonlinear meta-model can improve over Ridge stacking.

## Approach

**Stacking inputs:**
- catboost_oof_seedavg
- lgb_te_oof_seedavg
- Optional: cat - lgb, abs(cat - lgb)

**LightGBM stacker params (very small, heavily regularized):**
```python
{
    'num_leaves': 7,
    'max_depth': 3,
    'min_data_in_leaf': 2000,
    'lambda_l2': 1.0
}
```

## Results

| Model | CV RMSE |
|-------|---------|
| CatBoost (3 seeds) | 8.7701 |
| LightGBM TE (3 seeds) | 8.7607 |
| Ridge Stacker | 8.7558 |
| LGB Stacker (2 features) | 8.7586 |
| LGB Stacker (4 features) | 8.7586 |

**Delta (Ridge - LGB): -0.0028**
**Fold RMSE std: 0.0121**

### Comparison

| Experiment | CV RMSE | Diff |
|------------|---------|------|
| Ridge Stacker | 8.7558 | - |
| LGB Stacker | 8.7586 | -0.0028 |
| Exp 07 (best) | 8.7395 | - |

## Conclusion

**LGB meta-stacker is WORSE than Ridge by 0.0028.**

With only 2-4 stacking features, there's no nonlinear signal to exploit. Ridge is optimal for this linear combination task.

Decision: **Stop permanently** (improvement < 0.01).

## Usage

```bash
python3 train.py
```
