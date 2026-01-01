# Experiment 04: Simple MLP Neural Network

## Summary
| Model | CV RMSE |
|-------|---------|
| MLP (128-64-32) | **8.9394** |

Worse than tree ensemble (8.74), similar to polynomial regression (8.98).

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

## Results

### Per-Fold RMSE
| Fold | Best RMSE | Epochs |
|------|-----------|--------|
| 1 | 8.8745 | 81 (early stop) |
| 2 | 8.8875 | 51 (early stop) |
| 3 | 8.9155 | 38 (early stop) |
| 4 | 8.8905 | 56 (early stop) |
| 5 | 8.9855 | 51 (early stop) |

### Overall
- **CV RMSE: 8.9394**

## Training
- Device: MPS (Apple Silicon)
- 5-fold cross-validation
- Adam optimizer, lr=1e-3
- ReduceLROnPlateau scheduler
- MSE loss
- Early stopping (patience=10)
- Batch size: 1024
- Max epochs: 100

## Observations
- Simple MLP does **not** beat gradient boosting (8.94 vs 8.74)
- Performance is comparable to polynomial regression (8.98)
- No categorical embeddings used - just label-encoded integers
- Fold 5 had highest error (8.99) - some variance across folds

## Next Steps
- Add **categorical embeddings** to learn relationships between categories
- Extract embeddings and use as features in LightGBM
- Try deeper/wider architectures

## Files
- `train.py` - Main training script
- `submission_mlp.csv` - Predictions

## Run
```bash
cd experiments/04_neural_network
python train.py
```
