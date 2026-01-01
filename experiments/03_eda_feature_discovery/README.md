# Experiment 03: EDA & Feature Discovery

## Summary
Exploratory analysis to find patterns and feature ideas.

## Key Findings

### Strongest Correlations with exam_score
| Feature | Correlation |
|---------|-------------|
| **study_hours** | **0.76** |
| effort (study_hours × attendance) | 0.80 |
| class_attendance | 0.36 |
| sleep_score | 0.28 |
| sleep_hours | 0.17 |
| age | 0.01 (irrelevant) |

### Most Predictive Categories

**study_method** (biggest spread: 11.6 points)
| Method | Mean Score |
|--------|------------|
| coaching | 69.3 |
| mixed | 65.1 |
| group study | 60.5 |
| online videos | 59.7 |
| self-study | 57.7 |

**sleep_quality** (10.9 point spread)
| Quality | Mean Score |
|---------|------------|
| good | 67.9 |
| average | 62.7 |
| poor | 57.0 |

**facility_rating** (8.8 point spread)
| Rating | Mean Score |
|--------|------------|
| high | 66.7 |
| medium | 63.0 |
| low | 58.0 |

### Nearly Irrelevant Features
- **internet_access**: yes=62.51, no=62.48 (no difference!)
- **exam_difficulty**: easy/moderate/hard all ~62.5 (no difference!)
- **gender**: <1 point spread
- **course**: <1.5 point spread
- **age**: correlation 0.01

### High vs Low Scorers (key differences)
| Feature | High (>=80) | Low (<=40) | Diff |
|---------|-------------|------------|------|
| study_hours | 6.5 | 1.3 | **+5.2** |
| attendance | 80.8% | 60.6% | **+20%** |
| sleep_hours | 7.5 | 6.6 | +0.9 |
| coaching % | 33% | 10% | **+23%** |
| good sleep % | 48% | 19% | **+29%** |
| high facility % | 43% | 21% | **+22%** |

### Best New Feature
**effort = study_hours × class_attendance / 100**
- Correlation: **0.80** (higher than study_hours alone at 0.76!)

## Insights for Feature Engineering

1. **effort** is the strongest single feature - already using this
2. **study_method** and **sleep_quality** are highly predictive - ensure proper encoding
3. **internet_access** and **exam_difficulty** add almost no signal - could drop
4. **Interactions to try**:
   - study_method × study_hours (does coaching amplify study time?)
   - sleep_quality × study_hours (does good sleep amplify effort?)
   - facility_rating × attendance (do good facilities encourage attendance?)

## Files
- `eda.py` - Analysis script
- `README.md` - This file

## Run
```bash
cd experiments/03_eda_feature_discovery
python eda.py
```
