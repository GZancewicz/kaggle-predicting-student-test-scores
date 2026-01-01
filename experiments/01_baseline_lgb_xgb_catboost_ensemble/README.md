# Experiment 01: Baseline LGB + XGB + CatBoost Ensemble

## Description
Multi-model ensemble with hand-tuned hyperparameters and basic feature engineering.

## Models
- LightGBM (learning_rate=0.03, num_leaves=63, max_depth=8)
- XGBoost (learning_rate=0.03, max_depth=8)
- CatBoost (learning_rate=0.03, depth=8) - native categorical handling

## Feature Engineering (24 features total)
- **Numeric encodings** for ordinal categoricals (sleep_quality, facility_rating, etc.)
- **Interactions**: `study_efficiency = study_hours * attendance / 100`
- **Derived scores**: `sleep_score`, `resource_score`, `adjusted_study`
- **Binned features**: attendance and study_hours buckets
- **Combined**: `total_input`, `preparation_score`

## Training
- 5-fold cross-validation
- 2000 rounds with early stopping (patience=100)
- Optimal weight blending using scipy.optimize on OOF predictions

## Results

### Per-Fold RMSE

| Fold | LightGBM | XGBoost | CatBoost |
|------|----------|---------|----------|
| 1    | 8.7426   | 8.7424  | 17.4766  |
| 2    | 8.7481   | 8.7491  | 17.5106  |
| 3    | 8.7371   | 8.7383  | 17.5293  |
| 4    | 8.7566   | 8.7573  | 17.5657  |
| 5    | 8.7754   | 8.7733  | 17.5572  |

### Overall CV RMSE
- **LightGBM**: 8.7520
- **XGBoost**: 8.7521
- **CatBoost**: 17.5279 (broken - see notes)
- **Final Ensemble**: 8.7411

### Optimal Weights
- LightGBM: 50%
- XGBoost: 50%
- CatBoost: 0% (excluded due to poor performance)

## Notes
- **CatBoost issue**: RMSE ~17.5 indicates it was likely optimizing MSE internally and reporting a different metric, or there's a bug with categorical handling. The optimizer correctly assigned it 0% weight.
- **LGB vs XGB**: Nearly identical performance (8.752 vs 8.752)
- **Ensemble benefit**: Small improvement from 8.752 → 8.741 (0.01 RMSE)

## Files
- `train.py` - Main training script
- `submission_ensemble.csv` - Final submission (LGB+XGB blend)
- `submission_lgb.csv` - LightGBM only
- `submission_xgb.csv` - XGBoost only
- `submission_cat.csv` - CatBoost only (not recommended)

## Next Steps
- Fix CatBoost (likely needs numeric features, not raw categoricals with current params)
- Try target encoding
- Hyperparameter tuning with Optuna
- Add TabPFN to ensemble
