# Kaggle: Predicting Student Test Scores

**Playground Series - Season 6 Episode 1**

Competition to predict student exam scores based on study habits, demographics, and environmental factors.

## Results

| Experiment | Model | CV RMSE |
|------------|-------|---------|
| 01 | LightGBM + XGBoost Ensemble | **8.74** |
| 02 | Polynomial Ridge (degree 3) | 8.98 |
| 04 | MLP Neural Network | 8.94 |

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
│   └── 04_neural_network/
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
