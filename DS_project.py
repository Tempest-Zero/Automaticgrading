
"""Automatic Grading System


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
DEFAULT_ABSOLUTE_THRESHOLDS = {"A": 90, "B": 80, "C": 70, "D": 60}
DEFAULT_RELATIVE_DISTRIBUTION = {"A": 0.20, "B": 0.30, "C": 0.30, "D": 0.10, "F": 0.10}
PLOT_STYLE = dict(style="whitegrid", palette="pastel")

sns.set_theme(**PLOT_STYLE)


# -----------------------------------------------------------------------------
# Data loading & validation helpers
# -----------------------------------------------------------------------------

def load_scores(path: Path) -> pd.DataFrame:
    """Read *path* into a DataFrame and validate required columns.

    Expected columns: ``StudentID`` and ``Score`` (case‑sensitive).
    Raises a RuntimeError with a clear message for IO or schema issues.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"File not found: {path}") from exc
    except Exception as exc:  # noqa: BLE001  # broad but re‑raised with context
        raise RuntimeError(f"Failed to read CSV ({path}): {exc}") from exc

    required = {"StudentID", "Score"}
    if not required.issubset(df.columns):
        missing = ", ".join(required - set(df.columns))
        raise RuntimeError(f"CSV missing required column(s): {missing}")

    return df


# -----------------------------------------------------------------------------
# Absolute grading helpers
# -----------------------------------------------------------------------------

def grade_absolute(score: float, thresholds: dict[str, float]) -> str:
    """Return the letter grade for *score* using *thresholds*."""
    for letter, cutoff in (t for t in thresholds.items()):
        if score >= cutoff:
            return letter
    return "F"


# -----------------------------------------------------------------------------
# Relative grading helpers – normal‑curve
# -----------------------------------------------------------------------------

def attach_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with an ``AdjustedScore`` (z‑score) column."""
    mu = df["Score"].mean()
    sigma = df["Score"].std(ddof=0)

    if sigma == 0:  # All students have identical scores → nothing to scale
        return df.assign(AdjustedScore=df["Score"])

    z = (df["Score"] - mu) / sigma
    return df.assign(AdjustedScore=z)


def grade_from_percentiles(df: pd.DataFrame, *, output_col: str = "Grade") -> pd.DataFrame:
    """Map z‑score percentiles to letter grades and return new DataFrame."""
    if "AdjustedScore" not in df.columns:
        raise ValueError("Column 'AdjustedScore' missing – did you run attach_z_scores ?")

    p = norm.cdf(df["AdjustedScore"])  # percentile 0‑1
    bins = {"A": 0.80, "B": 0.50, "C": 0.20, "D": 0.10}

    def _map(percent: float) -> str:  # nested helper for clarity
        for letter, cutoff in bins.items():
            if percent >= cutoff:
                return letter
        return "F"

    return df.assign(**{output_col: p.map(_map)})


# -----------------------------------------------------------------------------
# Relative grading helpers – distribution buckets
# -----------------------------------------------------------------------------

def grade_from_distribution(
    df: pd.DataFrame,
    *,
    distribution: dict[str, float],
    output_col: str = "Grade",
) -> pd.DataFrame:
    """Assign letter grades strictly by position in sorted score list."""
    sorted_df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    n = len(sorted_df)

    counts = {g: round(pct * n) for g, pct in distribution.items()}
    # patch rounding drift on last grade bucket
    counts[list(counts.keys())[-1]] += n - sum(counts.values())

    labels = []
    start = 0
    for grade, cnt in counts.items():
        labels.extend([grade] * cnt)
        start += cnt

    sorted_df[output_col] = labels
    return sorted_df.sort_index()  # restore original index order


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_hist_kde(series: pd.Series, title: str, *, ax) -> None:  # noqa: D401
    """Histogram + KDE for *series* on *ax*."""
    sns.histplot(series, kde=True, ax=ax, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel(series.name)


def plot_grade_distribution(grades: pd.Series, *, title: str) -> None:
    sns.countplot(x=grades, order=sorted(grades.unique()), color="salmon")
    plt.title(title)
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: C901 – keep CLI glue in one place
    parser = argparse.ArgumentParser(description="Automatic grading system")
    parser.add_argument("--file", required=True, help="Path to CSV with StudentID,Score")
    parser.add_argument("--method", choices=["absolute", "relative"], default="relative")
    parser.add_argument(
        "--approach", choices=["normal-curve", "distribution"], default="normal-curve",
        help="Which relative grading approach to use",
    )
    args = parser.parse_args(argv)

    df_raw = load_scores(Path(args.file))

    # Basic statistics
    print("\nRAW SCORE STATS")
    print(df_raw["Score"].describe().to_string())

    # Plot raw distribution
    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    plot_hist_kde(df_raw["Score"], "Original Scores", ax=ax1)
    sns.kdeplot(df_raw["Score"], fill=True, ax=ax2, color="lightcoral")
    ax2.set_title("Density Plot of Original Scores")
    ax2.set_xlabel("Score")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Grade assignment
    # ------------------------------------------------------------------
    if args.method == "absolute":
        df_final = df_raw.assign(
            Grade=df_raw["Score"].apply(grade_absolute, thresholds=DEFAULT_ABSOLUTE_THRESHOLDS),
            AdjustedScore=df_raw["Score"],
        )
    else:  # relative
        if args.approach == "normal-curve":
            df_z = attach_z_scores(df_raw)
            df_final = grade_from_percentiles(df_z)
        else:
            df_final = grade_from_distribution(df_raw, distribution=DEFAULT_RELATIVE_DISTRIBUTION)
            df_final = df_final.assign(AdjustedScore=df_final["Score"])  # no transform

    # Display grade distribution
    print("\nFINAL GRADE COUNTS")
    print(df_final["Grade"].value_counts().sort_index())

    plot_grade_distribution(df_final["Grade"], title="Final Grade Distribution")

    # Export back to CSV for record‑keeping
    output_path = Path(args.file).with_stem(Path(args.file).stem + "_graded")
    df_final.to_csv(output_path, index=False)
    print(f"\nGraded file saved → {output_path.absolute()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        sys.exit(f"Error: {err}")
