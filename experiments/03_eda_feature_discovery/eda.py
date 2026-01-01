"""
Kaggle Playground Series S6E1: Predicting Student Test Scores
Experiment 03: EDA & Feature Discovery
"""

import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    return train, test


def basic_stats(df):
    """Basic statistics and distributions."""
    print("="*60)
    print("BASIC STATISTICS")
    print("="*60)

    print("\n--- Numeric Columns ---")
    print(df.describe().round(2))

    print("\n--- Target Distribution ---")
    print(f"Mean: {df['exam_score'].mean():.2f}")
    print(f"Std:  {df['exam_score'].std():.2f}")
    print(f"Min:  {df['exam_score'].min():.2f}")
    print(f"Max:  {df['exam_score'].max():.2f}")

    print("\n--- Categorical Value Counts ---")
    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())


def target_by_category(df):
    """Mean exam_score by each categorical variable."""
    print("\n" + "="*60)
    print("MEAN EXAM SCORE BY CATEGORY")
    print("="*60)

    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
                'study_method', 'facility_rating', 'exam_difficulty']

    for col in cat_cols:
        print(f"\n--- {col} ---")
        stats = df.groupby(col)['exam_score'].agg(['mean', 'std', 'count'])
        stats = stats.sort_values('mean', ascending=False)
        print(stats.round(2))


def numeric_correlations(df):
    """Correlation of numeric features with target."""
    print("\n" + "="*60)
    print("CORRELATIONS WITH EXAM_SCORE")
    print("="*60)

    numeric_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    for col in numeric_cols:
        corr = df[col].corr(df['exam_score'])
        print(f"{col}: {corr:.4f}")


def interaction_analysis(df):
    """Look for interesting interactions."""
    print("\n" + "="*60)
    print("INTERACTION ANALYSIS")
    print("="*60)

    # Course x Study Method
    print("\n--- Course × Study Method (mean score) ---")
    pivot = df.pivot_table(values='exam_score', index='course',
                           columns='study_method', aggfunc='mean')
    print(pivot.round(1))

    # Sleep Quality x Exam Difficulty
    print("\n--- Sleep Quality × Exam Difficulty (mean score) ---")
    pivot = df.pivot_table(values='exam_score', index='sleep_quality',
                           columns='exam_difficulty', aggfunc='mean')
    print(pivot.round(1))

    # Internet Access x Facility Rating
    print("\n--- Internet Access × Facility Rating (mean score) ---")
    pivot = df.pivot_table(values='exam_score', index='internet_access',
                           columns='facility_rating', aggfunc='mean')
    print(pivot.round(1))

    # Gender x Course
    print("\n--- Gender × Course (mean score) ---")
    pivot = df.pivot_table(values='exam_score', index='gender',
                           columns='course', aggfunc='mean')
    print(pivot.round(1))


def binned_analysis(df):
    """Analyze numeric features in bins."""
    print("\n" + "="*60)
    print("BINNED NUMERIC ANALYSIS")
    print("="*60)

    # Study hours bins
    print("\n--- Study Hours Bins ---")
    df['study_hours_bin'] = pd.cut(df['study_hours'], bins=[0, 2, 4, 6, 8, 10])
    print(df.groupby('study_hours_bin')['exam_score'].agg(['mean', 'std', 'count']).round(2))

    # Attendance bins
    print("\n--- Attendance Bins ---")
    df['attendance_bin'] = pd.cut(df['class_attendance'], bins=[0, 50, 70, 85, 95, 100])
    print(df.groupby('attendance_bin')['exam_score'].agg(['mean', 'std', 'count']).round(2))

    # Sleep hours bins
    print("\n--- Sleep Hours Bins ---")
    df['sleep_bin'] = pd.cut(df['sleep_hours'], bins=[4, 5, 6, 7, 8, 10])
    print(df.groupby('sleep_bin')['exam_score'].agg(['mean', 'std', 'count']).round(2))

    # Age bins
    print("\n--- Age Bins ---")
    print(df.groupby('age')['exam_score'].agg(['mean', 'std', 'count']).round(2))


def high_low_scorers(df):
    """Compare characteristics of high vs low scorers."""
    print("\n" + "="*60)
    print("HIGH vs LOW SCORERS")
    print("="*60)

    high = df[df['exam_score'] >= 80]
    low = df[df['exam_score'] <= 40]

    print(f"\nHigh scorers (>=80): {len(high)} students")
    print(f"Low scorers (<=40):  {len(low)} students")

    print("\n--- Numeric Means ---")
    numeric_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    comparison = pd.DataFrame({
        'High (>=80)': high[numeric_cols].mean(),
        'Low (<=40)': low[numeric_cols].mean(),
        'Diff': high[numeric_cols].mean() - low[numeric_cols].mean()
    })
    print(comparison.round(2))

    print("\n--- Category Distribution (%) ---")
    cat_cols = ['gender', 'course', 'sleep_quality', 'study_method',
                'facility_rating', 'exam_difficulty', 'internet_access']
    for col in cat_cols:
        print(f"\n{col}:")
        high_pct = (high[col].value_counts(normalize=True) * 100).round(1)
        low_pct = (low[col].value_counts(normalize=True) * 100).round(1)
        comp = pd.DataFrame({'High%': high_pct, 'Low%': low_pct})
        comp['Diff'] = comp['High%'] - comp['Low%']
        print(comp.sort_values('Diff', ascending=False))


def feature_ideas(df):
    """Test potential new features."""
    print("\n" + "="*60)
    print("POTENTIAL FEATURE IDEAS")
    print("="*60)

    # Create potential features
    df = df.copy()

    # Effort score
    df['effort'] = df['study_hours'] * df['class_attendance'] / 100

    # Sleep score (need to encode sleep_quality)
    sleep_map = {'poor': 1, 'average': 2, 'good': 3}
    df['sleep_quality_num'] = df['sleep_quality'].map(sleep_map)
    df['sleep_score'] = df['sleep_hours'] * df['sleep_quality_num']

    # Difficulty-adjusted effort
    diff_map = {'easy': 1, 'moderate': 2, 'hard': 3}
    df['difficulty_num'] = df['exam_difficulty'].map(diff_map)
    df['effort_per_difficulty'] = df['effort'] / df['difficulty_num']

    # Resource score
    internet_map = {'no': 0, 'yes': 1}
    facility_map = {'low': 1, 'medium': 2, 'high': 3}
    df['internet_num'] = df['internet_access'].map(internet_map)
    df['facility_num'] = df['facility_rating'].map(facility_map)
    df['resource_score'] = df['internet_num'] + df['facility_num']

    # Is STEM course
    df['is_stem'] = df['course'].isin(['b.tech', 'bca', 'b.sc']).astype(int)

    # Total preparation
    df['total_prep'] = df['effort'] * df['sleep_score'] * df['resource_score']

    # Correlations of new features
    new_features = ['effort', 'sleep_score', 'effort_per_difficulty',
                    'resource_score', 'is_stem', 'total_prep']

    print("\nCorrelation of new features with exam_score:")
    for feat in new_features:
        corr = df[feat].corr(df['exam_score'])
        print(f"  {feat}: {corr:.4f}")

    # Compare to original features
    print("\nOriginal feature correlations:")
    for col in ['study_hours', 'class_attendance', 'sleep_hours', 'age']:
        corr = df[col].corr(df['exam_score'])
        print(f"  {col}: {corr:.4f}")


def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")

    basic_stats(train)
    target_by_category(train)
    numeric_correlations(train)
    interaction_analysis(train)
    binned_analysis(train)
    high_low_scorers(train)
    feature_ideas(train)

    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
