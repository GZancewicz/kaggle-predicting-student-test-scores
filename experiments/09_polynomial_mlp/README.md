# Experiment 09: Polynomial Features MLP

MLP with full degree-2 polynomial expansion on custom-encoded features.

## Approach

Same custom encodings as Exp 08, plus:
- All features squared (x^2)
- All pairwise cross-terms (x1*x2, x1*x3, ...)

## Features

Starting with 11 base features:
- 4 numeric: age, study_hours, class_attendance, sleep_hours
- 7 encoded categoricals

After polynomial expansion:
- 11 original
- 11 squared
- 55 cross-terms (C(11,2))
- **77 total features**

## Model Architecture

- MLP: 256 -> 128 -> 64 -> 1 (wider first layer for more features)
- BatchNorm + ReLU + Dropout(0.2)
- 5-fold CV with early stopping

## Results (partial - aborted during fold 4)

| Fold | Best RMSE | Epochs |
|------|-----------|--------|
| 1 | 8.8963 | 62 (early stop) |
| 2 | 8.9142 | 59 (early stop) |
| 3 | 8.9155 | 67 (early stop) |

Estimated CV RMSE: ~8.91 (based on 3 folds)

Still underperforms gradient boosting (~8.74). Polynomial features don't help MLP.

## Usage

```bash
python3 train.py
```
