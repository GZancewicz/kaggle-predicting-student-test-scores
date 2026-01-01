# Experiment 05: Neural Network Embeddings + LightGBM

## Summary
| Model | CV RMSE |
|-------|---------|
| Embedding MLP alone | 8.9148 |
| LightGBM with embeddings | 8.7586 |
| Ensemble (MLP=0.3, LGB=0.7) | 8.7738 |

**Did not beat baseline (8.74).** Embeddings provided no improvement over standard LightGBM.

## Approach

1. Train MLP with learned categorical embeddings
2. Extract embeddings after training
3. Replace categorical features with their learned embedding vectors
4. Train LightGBM on expanded feature set

## Architecture

### Embedding Dimensions
| Feature | Categories | Embedding Dims |
|---------|------------|----------------|
| gender | 3 | 2 |
| course | 7 | 4 |
| study_method | 5 | 3 |
| sleep_quality | 3 | 2 |
| internet_access | 2 | 1 |
| facility_rating | 3 | 2 |
| exam_difficulty | 3 | 2 |
| **Total** | | **16 dims** |

### MLP Structure
```
Numeric (4) + Embeddings (16) = 20 input dims
    ↓
Linear(20 → 128) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(32 → 1)
```

## Results

### MLP Training (per fold)
| Fold | Best RMSE | Epochs |
|------|-----------|--------|
| 1 | 8.8797 | 41 |
| 2 | 8.8753 | 35 |
| 3 | 8.8932 | 27 |
| 4 | 8.8701 | 34 |
| 5 | 8.9406 | 24 |

### LightGBM with Embeddings (per fold)
| Fold | RMSE |
|------|------|
| 1 | 8.7474 |
| 2 | 8.7540 |
| 3 | 8.7445 |
| 4 | 8.7649 |
| 5 | 8.7819 |

## Why It Didn't Help

1. **Few categories**: Most features have only 2-7 categories - not enough complexity for embeddings to capture rich relationships
2. **LightGBM handles categoricals well**: Native categorical support in gradient boosting already learns effective splits
3. **Low-dimensional embeddings**: 16 total dims isn't much richer than one-hot encoding (26 dims)
4. **Simple relationships**: The data may have straightforward categorical effects that don't benefit from learned representations

## Learned Embeddings

Saved to `embeddings.json`. Notable patterns:

- **study_method**: self-study [1.04, 0.84, -0.46] and group study [1.01, 0.59, -0.11] cluster together; coaching [-0.53, 0.35, 0.74] is distinct
- **sleep_quality**: poor [-1.63, 0.76] has very negative first dim vs good [-0.02, 0.15]
- **course**: technical courses (b.tech, bca, diploma) have negative first dim; ba is positive

## Files
- `train.py` - Training script
- `embeddings.json` - Learned embedding vectors
- `submission_lgb_embeddings.csv` - LightGBM predictions
- `submission_mlp_embeddings.csv` - MLP predictions
- `submission_ensemble.csv` - Weighted ensemble

## Run
```bash
cd experiments/05_embedding_lgb
python3 train.py
```
