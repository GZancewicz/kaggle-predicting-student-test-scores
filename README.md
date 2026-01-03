# Kaggle: Predicting Student Test Scores

**Playground Series - Season 6 Episode 1**

Competition to predict student exam scores based on study habits, demographics, and environmental factors.

## Results

| Experiment | Model | CV RMSE | Summary |
|------------|-------|---------|---------|
| 01 | LightGBM + XGBoost Ensemble | 8.7411 | Baseline ensemble with hand-tuned params and basic feature engineering |
| 02 | Polynomial Ridge (degree 3) | 8.9818 | Linear model with polynomial feature expansion |
| 04 | MLP Neural Network | 8.9394 | Simple 3-layer neural network with label-encoded categoricals |
| 05 | Embeddings + LightGBM | 8.7586 | Learned categorical embeddings fed into LightGBM |
| 06 | Residual Correction | 8.7606 | Post-prediction adjustment based on residual slopes (no improvement) |
| 07 | Enhanced Features + LGB | **8.7395** | LightGBM with residual-informed engineered features |
| 08 | Custom Encoding MLP | 9.0296 | MLP with custom ordinal/frequency encodings for categoricals |
| 09 | Polynomial MLP | ~8.91 | MLP with degree-2 polynomial feature expansion (77 features) |
| 10 | Polynomial LGB/XGB Ensemble | 8.7568 | Exp 01 ensemble with polynomial features (worse than original) |
| 11 | Stacking Ensemble | 8.7604 | CatBoost + LGB with target encoding, Ridge stacker, calibration (1 seed) |
| 12 | Multi-Seed Stacking | 8.7563 | 3-seed averaging for variance reduction (minimal improvement) |
| 13 | Stratified Folds | 8.7607 | Stratified by study_hours bins (no improvement vs regular KFold) |
| 14 | Formula Discovery | 8.8885 | Residual analysis confirms data is NOT simple formula; trees capture nonlinear structure |
| 15 | LGB Meta-Stacker | 8.7586 | Nonlinear stacker worse than Ridge by 0.003; no signal in 2-4 stacking features |

**Best**: Experiment 07 (8.7395)
**Leaderboard target**: ~8.57

## Data

- **Train**: 630,000 students
- **Test**: 270,000 students
- **Target**: exam_score (19-100)
- **Metric**: RMSE

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | numeric | 17-24 |
| gender | categorical | male, female, other |
| course | categorical | b.tech, b.sc, b.com, bca, bba, ba, diploma |
| study_hours | numeric | 0-10 |
| class_attendance | numeric | 0-100% |
| internet_access | categorical | yes, no |
| sleep_hours | numeric | 4-10 |
| sleep_quality | categorical | poor, average, good |
| study_method | categorical | self-study, online videos, group study, coaching, mixed |
| facility_rating | categorical | low, medium, high |
| exam_difficulty | categorical | easy, moderate, hard |

### Key Insights (from EDA)

- **study_hours** is most predictive (correlation 0.76)
- **study_method** matters: coaching (69.3) vs self-study (57.7)
- **internet_access** and **exam_difficulty** have almost no signal

## Project Structure

```
├── data/                   # Competition data (gitignored)
├── doc/                    # Competition overview
├── experiments/            # Each modeling approach
│   ├── 01_baseline_lgb_xgb_catboost_ensemble/
│   ├── 02_polynomial_elasticnet/
│   ├── 03_eda_feature_discovery/
│   ├── 04_neural_network/
│   ├── 05_embedding_lgb/
│   ├── 06_residual_correction/
│   ├── 07_enhanced_features_embeddings/
│   ├── 08_custom_encoding_mlp/
│   ├── 09_polynomial_mlp/
│   ├── 10_ensemble_enhanced_features/
│   ├── 11_stacking_ensemble/
│   ├── 12_multiseed_stacking/
│   ├── 13_stratified_folds/
│   ├── 14_interaction_features/
│   └── 15_lgb_meta_stacker/
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

```bash
pip install -r requirements.txt
```

## Run an Experiment

```bash
cd experiments/01_baseline_lgb_xgb_catboost_ensemble
python train.py
```

## Competition Link

https://www.kaggle.com/competitions/playground-series-s6e1
