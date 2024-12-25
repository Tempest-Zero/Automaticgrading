import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

sns.set_theme(style="whitegrid", palette="pastel")

def read_csv_data(filepath):
    """
    Reads a CSV file containing at least two columns: 'StudentID' and 'Score'.
    Returns a pandas DataFrame or raises an Exception if the file is missing columns.
    """
    try:
        df = pd.read_csv(filepath)
        df.head
    except FileNotFoundError: #exception handling
        raise FileNotFoundError(f"Could not find the file: {filepath}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV: {e}")

    # Basic error handling for missing columns
    required_cols = {'StudentID', 'Score'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    return df


def assign_absolute_grade(score, thresholds=None):
    """
    Assigns an absolute letter grade based on fixed numeric thresholds.
    Example thresholds:
    {
        'A': 90,
        'B': 80,
        'C': 70,
        'D': 60
    }
    """
    if thresholds is None:
        thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}

    if score >= thresholds['A']:
        return 'A'
    elif score >= thresholds['B']:
        return 'B'
    elif score >= thresholds['C']:
        return 'C'
    elif score >= thresholds['D']:
        return 'D'
    else:
        return 'F'


def transform_scores_normal_curve(df):
    """
    Performs z-score scaling (normal-curve approach) on 'Score'.
    1) Compute mean and std of the scores
    2) Convert each score to a z-score
    3) Rescale z-scores to e.g. 0-100 range (optional) or keep them as z-scores
    Returns a DataFrame with a new column 'AdjustedScore' reflecting the transformation.
    """
    df_new = df.copy()
    mu = df['Score'].mean()
    sigma = df['Score'].std()

    if sigma == 0:
        # All scores are identical; no transformation
        df_new['AdjustedScore'] = df['Score']  # or just keep them the same
        return df_new

    # Standard z-score
    z_scores = (df['Score'] - mu) / sigma

    # Optionally, you could map z-scores to some 0-100 scale. For demonstration:
    #   z_min, z_max = z_scores.min(), z_scores.max()
    #   z_range = z_max - z_min
    #   AdjustedScore = (z_scores - z_min) / z_range * 100
    #
    # But often we just keep the z-score as is, or keep the original scores for letter assignment.
    # We'll do a simple approach: keep z-scores in a new column
    df_new['AdjustedScore'] = z_scores
    return df_new


def assign_letter_grades_from_percentiles(df, grade_col='FinalGrade'):
    """
    Assign letter grades based on the percentile of 'AdjustedScore' in a normal distribution.
    By default:
      - A: top 20% (percentile >= 0.80)
      - B: next 30% (0.50 to 0.80)
      - C: next 30% (0.20 to 0.50)
      - D: next 10% (0.10 to 0.20)
      - F: bottom 10% (0.00 to 0.10)
    Stores the letter grade in df[grade_col].
    """
    df_new = df.copy()
    # We assume 'AdjustedScore' is a z-score. Use the normal CDF to get percentile
    if 'AdjustedScore' not in df_new.columns:
        # If somehow we didn't do a transform, just copy original scores for percentile
        df_new['AdjustedScore'] = df_new['Score']

    # If standard deviation was 0, everyone is the same, so let's handle that:
    if df_new['AdjustedScore'].nunique() == 1:
        df_new[grade_col] = 'C'
        return df_new

    z_scores = df_new['AdjustedScore']
    percentiles = norm.cdf(z_scores)  # ranges from 0 to 1

    # Default bins
    letter_bins = {
        'A': 0.80,
        'B': 0.50,
        'C': 0.20,
        'D': 0.10,
        'F': 0.00
    }
    letter_grades = []
    for p in percentiles:
        if p >= letter_bins['A']:
            letter_grades.append('A')
        elif p >= letter_bins['B']:
            letter_grades.append('B')
        elif p >= letter_bins['C']:
            letter_grades.append('C')
        elif p >= letter_bins['D']:
            letter_grades.append('D')
        else:
            letter_grades.append('F')

    df_new[grade_col] = letter_grades
    return df_new


def assign_relative_grade_distribution(df, distribution=None, grade_col='FinalGrade'):
    """
    Forces letter grades by sorting from highest to lowest Score and assigning
    top X% to 'A', next Y% to 'B', etc., based on the given distribution.
    """
    df_new = df.copy()

    if distribution is None:
        distribution = {'A': 0.20, 'B': 0.30, 'C': 0.30, 'D': 0.10, 'F': 0.10}

    # Sort scores descending
    sorted_df = df_new.sort_values(by='Score', ascending=False).reset_index(drop=True)
    n = len(sorted_df)

    # Convert percentages to counts
    grade_counts = {}
    for g, pct in distribution.items():
        grade_counts[g] = int(round(pct * n))

    # Correct rounding errors
    diff = n - sum(grade_counts.values())
    if diff != 0:
        last_grade = list(distribution.keys())[-1]
        grade_counts[last_grade] += diff

    # Assign
    assigned_grades = [''] * n
    start_idx = 0
    for g in distribution.keys():
        count = grade_counts[g]
        end_idx = start_idx + count
        for i in range(start_idx, end_idx):
            assigned_grades[i] = g
        start_idx = end_idx

    sorted_df[grade_col] = assigned_grades

    # Merge back with original order
    df_merged = pd.merge(df_new, sorted_df[['StudentID', grade_col]], on='StudentID', how='left')
    return df_merged



def main():
    filepath = "/content/StudentScores.csv"  # update if needed
    df = read_csv_data(filepath)

    # Instructor choice: 'absolute' or 'relative'
    grading_method = 'relative'

    # If 'relative', choose approach:
    #  - 'normal_curve' : z-score scaling, then assign letter grades by percentile
    #  - 'distribution' : direct percentage-based assignment
    relative_approach = 'normal_curve'

    # Example of absolute thresholds
    abs_thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}

    # Example of relative distribution
    rel_distribution = {'A': 0.20, 'B': 0.30, 'C': 0.30, 'D': 0.10, 'F': 0.10}


    mean_score = df['Score'].mean()
    var_score = df['Score'].var()
    std_score = df['Score'].std()
    skew_score = df['Score'].skew()

    print("\n=== ORIGINAL SCORES STATISTICS ===")
    print(f"Mean Score         : {mean_score:.2f}")
    print(f"Variance           : {var_score:.2f}")
    print(f"Standard Deviation : {std_score:.2f}")
    print(f"Skewness           : {skew_score:.2f}\n")

    # Plot histogram and density of original scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    sns.histplot(df['Score'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Histogram of Original Scores", fontsize=14)
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")

    # Density plot
    sns.kdeplot(df['Score'], fill=True, ax=axes[1], color='lightcoral')
    axes[1].set_title("Density Plot of Original Scores", fontsize=14)
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


    df_grading = df.copy()

    if grading_method.lower() == 'absolute':
        # =========== ABSOLUTE GRADING ===========
        df_grading['Grade'] = df_grading['Score'].apply(
            assign_absolute_grade, thresholds=abs_thresholds
        )
        # No numeric transformation, so 'AdjustedScore' = original
        df_grading['AdjustedScore'] = df_grading['Score']

    else:
        # =========== RELATIVE GRADING ===========
        if relative_approach == 'normal_curve':
            # 1) Transform scores to z-scores (or scaled scores)
            df_transformed = transform_scores_normal_curve(df_grading)
            # 2) Assign letter grades from the new z-score distribution
            df_final = assign_letter_grades_from_percentiles(df_transformed, grade_col='Grade')
            df_grading = df_final.copy()

        else:
            # Distribution-based approach
            df_final = assign_relative_grade_distribution(
                df_grading,
                distribution=rel_distribution,
                grade_col='Grade'
            )
            # In this approach, we did not transform the numeric scores
            df_final['AdjustedScore'] = df_final['Score']
            df_grading = df_final.copy()


    # Show the final distribution of assigned grades
    final_counts = df_grading['Grade'].value_counts().sort_index()
    print(f"=== FINAL GRADE DISTRIBUTION ({grading_method.upper()}) ===")
    for g, cnt in final_counts.items():
        print(f"Grade {g}: {cnt} students")
    print()

    # Bar chart of final grades
    plt.figure(figsize=(8, 5))
    sns.barplot(x=final_counts.index, y=final_counts.values, color='salmon')
    plt.title(f"Final Grade Distribution ({grading_method.capitalize()})", fontsize=14)
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.show()

    # This is most relevant if we used a normal-curve approach (z-scores or scaled).
    if 'AdjustedScore' in df_grading.columns:
        print("=== Adjusted Scores Analysis (for Relative Grading) ===")
        adj_mean = df_grading['AdjustedScore'].mean()
        adj_std = df_grading['AdjustedScore'].std()

        print(f"Adjusted Mean: {adj_mean:.2f}")
        print(f"Adjusted Std : {adj_std:.2f}\n")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        sns.histplot(df_grading['AdjustedScore'], kde=True, ax=axes[0], color='blueviolet')
        axes[0].set_title("Histogram of Adjusted Scores", fontsize=14)
        axes[0].set_xlabel("AdjustedScore")
        axes[0].set_ylabel("Count")

        # Density
        sns.kdeplot(df_grading['AdjustedScore'], fill=True, ax=axes[1], color='goldenrod')
        axes[1].set_title("Density Plot of Adjusted Scores", fontsize=14)
        axes[1].set_xlabel("AdjustedScore")
        axes[1].set_ylabel("Density")

        plt.tight_layout()
        plt.show()


    # Let's see how many changed letter grades if we compare to absolute thresholds:
    df_abs_compare = df.copy()
    df_abs_compare['AbsGrade'] = df_abs_compare['Score'].apply(
        assign_absolute_grade, thresholds=abs_thresholds
    )
    merged = pd.merge(
        df_abs_compare[['StudentID', 'AbsGrade']], 
        df_grading[['StudentID', 'Grade']], 
        on='StudentID', 
        how='left'
    )
    merged['Changed'] = merged['AbsGrade'] != merged['Grade']
    changed_count = merged['Changed'].sum()

    print(f"Number of students who changed letter grade compared to absolute grading: {changed_count}\n")
    print("Process completed successfully!")
    print("Feel free to modify thresholds, distributions, or methods.")
    print(" - Switch between 'absolute' and 'relative' in grading_method.")
    print(" - For 'relative', choose 'normal_curve' or 'distribution' in relative_approach.\n")


if _name_ == '_main_':
    main()
