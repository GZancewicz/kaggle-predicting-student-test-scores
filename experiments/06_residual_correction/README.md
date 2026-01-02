# Experiment 06: Residual Correction

Post-prediction adjustment on Experiment 05 predictions.

## Approach

Two methods to correct systematic biases detected in residual analysis:

1. **Simple Linear Correction**: Apply linear adjustment based on feature slopes
2. **Stacked Correction**: Train Ridge regression to predict residuals

## Key Findings

- Residual slopes from Exp 05 were small (0.007-0.154)
- Corrections provided minimal improvement
- Model is already well-calibrated

## Results

| Method | CV RMSE | Change |
|--------|---------|--------|
| Baseline (Exp 05) | 8.7586 | - |
| Best Correction | 8.7606 | +0.002 |

No improvement - model was already well-calibrated.

## Usage

```bash
python3 train.py
```

Requires Exp 05 outputs: `oof_lgb_emb.npy`, `y_train.npy`, `submission_lgb_embeddings.csv`
