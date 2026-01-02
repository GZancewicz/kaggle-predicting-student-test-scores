# Experiment 12: Multi-Seed Stacking

Variance reduction via 3-seed averaging, then re-stack. No feature changes, no fold changes, no hyperparameter tuning.

## Goal

Validate whether variance reduction via multi-seed averaging unlocks the next performance tier before touching anything else.

## Components

1. **CatBoost** - native categoricals + ordered boosting (3 seeds)
2. **LightGBM GOSS** - OOF target encoding (3 seeds)
3. **Ridge stacker** - on seed-averaged predictions
4. **Linear calibration** + clipping to [19, 100]

## Configuration

- N_FOLDS = 5 (same as Exp 11)
- SEEDS = [42, 202, 999]
- TARGET_SMOOTHING = 10
- CLIP_RANGE = [19, 100]

## Methodology

1. Train each model across all folds × all seeds
2. Collect OOF predictions per seed
3. Average OOF/test predictions across seeds
4. Stack seed-averaged predictions with Ridge
5. Apply linear calibration + clipping

## Results

*Run `python3 train.py` to populate*

## Decision Rule

- If CV improves >= 0.03 vs Exp 11 -> Scale to 5-10 seeds
- If CV improves < 0.02 -> Stop scaling, revisit fold strategy or features

## Usage

```bash
python3 train.py
```

## Outputs

- `submission_catboost_seedavg.csv`
- `submission_lgb_te_seedavg.csv`
- `submission_stacked_seedavg.csv`
- `submission_final.csv`
