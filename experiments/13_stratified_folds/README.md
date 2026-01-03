# Experiment 13: Stratified Folds by Binned study_hours

Test whether CV splits are mismatched to test distribution by stratifying folds on study_hours.

## Approach

- StratifiedKFold instead of KFold
- Stratify by study_hours binned into 10 quantiles
- Everything else identical to Exp 11 (1 seed, same params)

## Results

| Model | CV RMSE |
|-------|---------|
| CatBoost (Ordered) | 8.7748 |
| LightGBM TE (GOSS) | 8.7720 |
| Simple Average | 8.7612 |
| Ridge Stacker | 8.7610 |
| **Final (clipped)** | **8.7607** |

### Comparison

| Experiment | CV RMSE | Diff vs Exp 11 |
|------------|---------|----------------|
| Exp 11 (regular KFold) | 8.7604 | - |
| **Exp 13 (stratified)** | 8.7607 | -0.0003 |
| Exp 07 (best) | 8.7395 | - |

**No improvement** - stratified folds made essentially no difference (-0.0003).

## Conclusion

CV fold strategy is not the issue. The regular KFold splits are already well-distributed.

## Runtime

- CatBoost: 34.8 min
- LightGBM: 2.8 min
- Total: 37.6 min

## Usage

```bash
python3 train.py
```
