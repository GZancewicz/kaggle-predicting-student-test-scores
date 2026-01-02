# Experiment 10: LGB/XGB Ensemble with Polynomial Features

Combines Exp 01's ensemble approach with Exp 09's polynomial features.

## Approach

- LightGBM + XGBoost ensemble (from Exp 01)
- Polynomial features: custom encodings + squared + cross-terms (from Exp 09)
- 77 total features

## Features

Base (11):
- 4 numeric: age, study_hours, class_attendance, sleep_hours
- 7 custom encoded categoricals

After polynomial expansion (77):
- 11 original
- 11 squared
- 55 cross-terms

## Results

| Model | CV RMSE |
|-------|---------|
| LightGBM | 8.7691 |
| XGBoost | 8.7644 |
| **Ensemble** | **8.7568** |

Optimal weights: LGB 44%, XGB 56%

### Comparison

| Experiment | CV RMSE | Diff |
|------------|---------|------|
| Exp 01 (original) | 8.7411 | - |
| Exp 07 (best) | 8.7395 | -0.0016 |
| **Exp 10 (poly)** | 8.7568 | +0.0157 |

Polynomial features did NOT help - worse than original Exp 01 by 0.016 RMSE.

## Usage

```bash
python3 train.py
```
