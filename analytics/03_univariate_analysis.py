"""
Analytics 2.3 — Titanic univariate analysis.

This stage reads the cleaned dataset from Stage 2.2 and performs:
- Histograms for age and fare
- Box plots for age and fare
- IQR outlier counts for age and fare
- Fare mean, median, and mode
- A skewness conclusion based on mean/median/mode ordering

No dataset loading from Seaborn occurs here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "titanic_cleaned_stage_2_2.csv"
PLOTS_DIR = BASE_DIR / "plots" / "stage_2_3"


def iqr_outlier_count(series: pd.Series) -> tuple[float, float, float, float, int]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)]

    return q1, q3, lower_bound, upper_bound, len(outliers)


def save_histogram(series: pd.Series, column_name: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=30)
    plt.title(f"{column_name.capitalize()} Distribution")
    plt.xlabel(column_name.capitalize())
    plt.ylabel("Frequency")
    plt.tight_layout()

    path = PLOTS_DIR / f"{column_name}_histogram.png"
    plt.savefig(path, dpi=150)
    plt.close()


def save_boxplot(series: pd.Series, column_name: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.boxplot(series.dropna(), vert=False)
    plt.title(f"{column_name.capitalize()} Box Plot")
    plt.xlabel(column_name.capitalize())
    plt.tight_layout()

    path = PLOTS_DIR / f"{column_name}_boxplot.png"
    plt.savefig(path, dpi=150)
    plt.close()


def fare_skewness_conclusion(
    mean: float, median: float, mode: float
) -> str:
    if mean > median > mode:
        return (
            "Fare is positively (right) skewed because mean > median > mode."
        )
    elif mean < median < mode:
        return (
            "Fare is negatively (left) skewed because mean < median < mode."
        )
    else:
        return (
            "Fare does not follow a clear skewness pattern from the "
            "mean/median/mode ordering alone."
        )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_PATH}\n"
            "Run Stage 2.2 first."
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.3: UNIVARIATE ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Age analysis
    # ------------------------------------------------------------------
    age = df["age"].dropna()

    age_q1, age_q3, age_lower, age_upper, age_outliers = iqr_outlier_count(
        age
    )

    print("\n--- AGE IQR OUTLIER ANALYSIS ---")
    print(f"Q1: {age_q1:.4f}")
    print(f"Q3: {age_q3:.4f}")
    print(f"IQR: {age_q3 - age_q1:.4f}")
    print(f"Lower bound: {age_lower:.4f}")
    print(f"Upper bound: {age_upper:.4f}")
    print(f"Outlier count: {age_outliers}")

    save_histogram(age, "age")
    save_boxplot(age, "age")

    # ------------------------------------------------------------------
    # Fare analysis
    # ------------------------------------------------------------------
    fare = df["fare"].dropna()

    fare_q1, fare_q3, fare_lower, fare_upper, fare_outliers = iqr_outlier_count(
        fare
    )

    fare_mean = fare.mean()
    fare_median = fare.median()
    fare_modes = fare.mode()
    fare_mode = fare_modes.iloc[0]

    print("\n--- FARE SUMMARY ---")
    print(f"Mean: {fare_mean:.4f}")
    print(f"Median: {fare_median:.4f}")
    print(f"Mode: {fare_mode:.4f}")

    print("\n--- FARE IQR OUTLIER ANALYSIS ---")
    print(f"Q1: {fare_q1:.4f}")
    print(f"Q3: {fare_q3:.4f}")
    print(f"IQR: {fare_q3 - fare_q1:.4f}")
    print(f"Lower bound: {fare_lower:.4f}")
    print(f"Upper bound: {fare_upper:.4f}")
    print(f"Outlier count: {fare_outliers}")

    print("\n--- FARE SKEWNESS CONCLUSION ---")
    print(fare_skewness_conclusion(fare_mean, fare_median, fare_mode))

    save_histogram(fare, "fare")
    save_boxplot(fare, "fare")

    print("\n--- SAVED PLOTS ---")
    print(PLOTS_DIR / "age_histogram.png")
    print(PLOTS_DIR / "age_boxplot.png")
    print(PLOTS_DIR / "fare_histogram.png")
    print(PLOTS_DIR / "fare_boxplot.png")

    print("\nStage 2.3 complete.")


if __name__ == "__main__":
    main()
