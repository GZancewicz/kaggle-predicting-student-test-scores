# Experiment 12: Multi-Seed Stacking

Variance reduction via 3-seed averaging, then re-stack. No feature changes, no fold changes.

## Results

| Model | CV RMSE |
|-------|---------|
| CatBoost (3 seeds) | 8.7701 |
| LightGBM TE (3 seeds) | 8.7612 |
| Simple Average | 8.7573 |
| Ridge Stacker | 8.7565 |
| **Final (clipped)** | **8.7563** |

### Per-Seed Results

**CatBoost**: [8.7741, 8.7716, 8.7750] (std: 0.0014)
**LightGBM**: [8.7707, 8.7709, 8.7713] (std: 0.0003)

### Comparison

| Experiment | CV RMSE | Diff |
|------------|---------|------|
| Exp 11 (1 seed) | 8.7604 | - |
| **Exp 12 (3 seeds)** | 8.7563 | +0.0041 |
| Exp 07 (best) | 8.7395 | - |

**Minimal improvement** (+0.0041) from multi-seed averaging. Variance is very low - not the bottleneck.

## Conclusion

- Seed std is tiny (CatBoost: 0.0014, LGB: 0.0003)
- Scaling to more seeds won't help
- Issue is signal, not variance

## Runtime

- CatBoost: 104.6 min (3 seeds)
- LightGBM: 8.5 min (3 seeds)
- Total: 113.2 min

## Usage

```bash
python3 train.py
```
