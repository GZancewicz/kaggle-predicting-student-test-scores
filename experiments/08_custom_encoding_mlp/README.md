# Experiment 08: Custom Encoding MLP

PyTorch MLP with custom categorical encodings.

## Categorical Encodings

| Feature | Encoding |
|---------|----------|
| gender | female=0, other=2, male=3 |
| course | frequency encoding (% of total) |
| internet_access | no=0, yes=1 |
| sleep_quality | poor=0, average=1, good=3 |
| study_method | frequency encoding (% of total) |
| facility_rating | low=0, medium=1, high=2 |
| exam_difficulty | easy=0, moderate=1, hard=2 |

## Features

17 total features:
- 4 base numeric: age, study_hours, class_attendance, sleep_hours
- 7 custom encoded categoricals
- 6 engineered: study_hours_sq, study_hours_x_difficulty, sleep_score, resource_score, study_efficiency, low_attendance

## Model Architecture

- MLP: 128 -> 64 -> 32 -> 1
- BatchNorm + ReLU + Dropout(0.2)
- 5-fold CV with early stopping

## Usage

```bash
python3 train.py
```
