# Experiment 07: Enhanced Features + Embeddings

LightGBM with new engineered features based on residual analysis.

## Approach

Added new features targeting patterns found in residual analysis:
- `study_hours_sq`: Quadratic term for study hours
- `study_hours_x_difficulty`: Interaction term
- `sleep_score`: sleep_hours * sleep_quality_num
- `resource_score`: facility * internet_access
- `low_attendance`: Binary flag for attendance < 60%
- `study_efficiency`: study_hours * attendance / 100
- `adjusted_study`: study_hours / difficulty
- `total_preparation`: Combined preparation metric

## Features

- 12 numeric features (4 base + 8 engineered)
- 7 categorical features (label encoded)

## Usage

```bash
python3 train.py
```

Requires Exp 05 outputs for comparison.
