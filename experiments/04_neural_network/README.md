# Experiment 04: Simple MLP Neural Network

## Summary
| Model | CV RMSE |
|-------|---------|
| MLP (128-64-32) | TBD |

## Description
Simple 3-layer Multi-Layer Perceptron using PyTorch.

## Architecture
```
Input (11 features)
    ↓
Linear(11 → 128) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(32 → 1)
    ↓
Output (exam_score)
```

## Training
- 5-fold cross-validation
- Adam optimizer, lr=1e-3
- ReduceLROnPlateau scheduler
- MSE loss
- Early stopping (patience=10)
- Batch size: 1024
- Max epochs: 100

## Run
```bash
cd experiments/04_neural_network
python train.py
```
