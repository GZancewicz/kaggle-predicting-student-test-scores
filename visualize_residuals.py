"""
Visualize residual errors vs each feature.

Usage: python3 visualize_residuals.py <experiment_folder> [oof_file]
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def find_oof_file(folder):
    """Find the first oof_*.npy file in folder."""
    pattern = os.path.join(folder, 'oof_*.npy')
    files = glob.glob(pattern)
    if not files:
        return None
    for f in files:
        if 'ensemble' not in f:
            return f
    return files[0]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_residuals.py <experiment_folder> [oof_file]")
        print("\nAvailable experiments:")
        for folder in sorted(glob.glob('experiments/*/')):
            oof_files = glob.glob(os.path.join(folder, 'oof_*.npy'))
            if oof_files:
                names = [os.path.basename(f) for f in oof_files]
                print(f"  {folder}: {', '.join(names)}")
        sys.exit(1)

    folder = sys.argv[1]

    if len(sys.argv) >= 3:
        oof_file = os.path.join(folder, sys.argv[2])
    else:
        oof_file = find_oof_file(folder)
        if not oof_file:
            print(f"No oof_*.npy files found in {folder}")
            sys.exit(1)

    y_train_file = os.path.join(folder, 'y_train.npy')

    if not os.path.exists(oof_file):
        print(f"OOF file not found: {oof_file}")
        sys.exit(1)
    if not os.path.exists(y_train_file):
        print(f"y_train.npy not found in {folder}")
        sys.exit(1)

    # Load predictions
    print(f"Loading: {oof_file}")
    y_train = np.load(y_train_file)
    oof_preds = np.load(oof_file)
    residuals = y_train - oof_preds
    rmse = np.sqrt(np.mean(residuals**2))

    # Load original training data for features
    train_df = pd.read_csv('data/train.csv')

    # All feature columns
    feature_cols = ['age', 'gender', 'course', 'study_hours', 'class_attendance',
                    'internet_access', 'sleep_hours', 'sleep_quality',
                    'study_method', 'facility_rating', 'exam_difficulty']

    model_name = os.path.basename(oof_file).replace('oof_', '').replace('.npy', '')

    # Create subplots - 3 rows x 4 cols
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'{model_name} Residuals by Feature (RMSE: {rmse:.4f})', fontsize=14)
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        feature_vals = train_df[col]

        if train_df[col].dtype == 'object' or train_df[col].nunique() <= 10:
            # Categorical: boxplot of residuals per category, ordered by mean error
            categories = list(train_df[col].unique())
            mean_errors = {cat: residuals[train_df[col] == cat].mean() for cat in categories}
            categories_sorted = sorted(categories, key=lambda x: mean_errors[x])

            box_data = [residuals[train_df[col] == cat] for cat in categories_sorted]
            labels = [f"{cat}\n({mean_errors[cat]:.2f})" for cat in categories_sorted]
            ax.boxplot(box_data, labels=labels)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

            # Fit regression line through category means (x = position 1,2,3..., y = mean error)
            x_positions = np.arange(1, len(categories_sorted) + 1)
            y_means = np.array([mean_errors[cat] for cat in categories_sorted])
            slope, intercept, _, _, _ = stats.linregress(x_positions, y_means)
            ax.plot(x_positions, slope * x_positions + intercept, 'r-', linewidth=2, label=f'slope={slope:.3f}')
            ax.legend(fontsize=8)

            ax.set_xlabel(col)
            ax.set_ylabel('Residual')
            ax.tick_params(axis='x', rotation=45, labelsize=7)
        else:
            # Numeric: scatter plot with regression line
            ax.scatter(feature_vals, residuals, alpha=0.2, s=5)

            # Fit and plot regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(feature_vals, residuals)
            x_line = np.array([feature_vals.min(), feature_vals.max()])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'slope={slope:.3f}')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlabel(col)
            ax.set_ylabel('Residual')

        ax.set_title(col)

    # Hide unused subplot
    axes[-1].axis('off')

    plt.tight_layout()
    output_file = f'residuals_by_feature_{model_name}.png'
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f"Saved {output_file}")


if __name__ == '__main__':
    main()
