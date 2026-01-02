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

## Usage

```bash
python3 train.py
```
