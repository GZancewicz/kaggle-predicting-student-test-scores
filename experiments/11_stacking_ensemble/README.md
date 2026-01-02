# Experiment 11: Stacking Ensemble with Target Encoding

Comprehensive stacking approach based on GPT recommendations.

## Components

1. **CatBoost** with native categorical handling + ordered boosting
2. **LightGBM GOSS** with OOF target encoding + frequency encoding
3. **Ridge stacker** on OOF predictions (optionally with study_hours/attendance)
4. **Linear calibration** (y = a*pred + b fitted on OOF)
5. **Clipping** to [19, 100]

## Key Improvements

- **Proper CatBoost**: Native categorical handling with ordered boosting (Exp 01 CatBoost was broken)
- **OOF-safe target encoding**: No leakage, smoothing with m=10, noise=0
- **Multi-seed averaging**: Reduces variance (scalable from 1 to 10 seeds)
- **Stacking**: Learns optimal combination weights

## Target Encoding

```
enc = (sum_y + m * global_mean) / (count + m)
```

With m=10 smoothing, no noise.

## Configuration

- N_FOLDS = 5
- CATBOOST_SEEDS = [42] (scalable to [42, 202, 999])
- LGB_SEEDS = [42] (scalable to [42, 202, 999])
- TARGET_SMOOTHING = 10 (tunable: 5, 10, 20, 50)
- CLIP_RANGE = [19, 100]

## Results (1 seed each)

| Model | CV RMSE |
|-------|---------|
| CatBoost (Ordered) | 8.7741 |
| LightGBM TE (GOSS) | 8.7707 |
| Simple Average | 8.7609 |
| **Final (stacked+cal)** | **8.7604** |

### Comparison

| Experiment | CV RMSE | Diff |
|------------|---------|------|
| Exp 07 (best) | 8.7395 | - |
| Exp 01 (baseline) | 8.7411 | +0.0016 |
| **Exp 11 (stacking)** | 8.7604 | +0.0209 |

With only 1 seed, stacking does NOT beat Exp 07. CatBoost ordered boosting is working properly (8.77 vs 17.5 broken in Exp 01).

## Next Steps

- Scale to 3 seeds to reduce variance
- Try different smoothing values (m=5, 20, 50)
- Add more base models (XGBoost, different LGB configs)

## Usage

```bash
python3 train.py
```

## Outputs

- `oof_catboost_mean.npy` - CatBoost mean OOF
- `oof_lgb_te_mean.npy` - LightGBM TE mean OOF
- `oof_final.npy` - Final stacked + calibrated OOF
- `submission_final.csv` - Best submission for Kaggle
