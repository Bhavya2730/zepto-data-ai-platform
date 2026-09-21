"""
Analytics 2.6 — EDA-stage standardization of age and fare.

This stage standardizes age and fare on the FULL cleaned DataFrame from
Stage 2.2. This is exploratory-analysis standardization only.

Important:
- This is NOT the modeling preprocessing step.
- Modeling preprocessing must be fitted on the training split only.
- The cleaned Stage 2.2 dataset is used as the input.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "titanic_cleaned_stage_2_2.csv"
OUTPUT_PATH = BASE_DIR / "titanic_eda_standardized.csv"


def summarize(series: pd.Series) -> dict[str, float]:
    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_PATH}\n"
            "Run Stage 2.2 first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = ["age", "fare"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing from input: {missing_columns}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Stage 2.2 should have handled missing values in age/fare "
            "before standardization."
        )

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.6: EDA STANDARDIZATION")
    print("=" * 80)

    print("\n--- BEFORE STANDARDIZATION ---")

    before = pd.DataFrame(
        {
            "variable": ["age", "fare"],
            "mean": [df["age"].mean(), df["fare"].mean()],
            "std": [df["age"].std(), df["fare"].std()],
        }
    )

    print(before.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Standardize on the full cleaned DataFrame as required for EDA.
    scaler = StandardScaler()
    df[["age_standardized", "fare_standardized"]] = scaler.fit_transform(
        df[["age", "fare"]]
    )

    print("\n--- AFTER STANDARDIZATION ---")

    after = pd.DataFrame(
        {
            "variable": ["age_standardized", "fare_standardized"],
            "mean": [
                df["age_standardized"].mean(),
                df["fare_standardized"].mean(),
            ],
            "std": [
                df["age_standardized"].std(),
                df["fare_standardized"].std(),
            ],
        }
    )

    print(after.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n--- VERIFICATION ---")

    for column in ["age_standardized", "fare_standardized"]:
        mean = df[column].mean()
        std = df[column].std()

        print(
            f"{column}: mean={mean:.6f}, std={std:.6f}"
        )

        if abs(mean) < 1e-10:
            print("  Mean is approximately 0: PASS")
        else:
            print("  Mean is approximately 0: CHECK")

        # pandas std() uses sample standard deviation (ddof=1), whereas
        # StandardScaler uses population standard deviation (ddof=0).
        # Therefore the pandas value is expected to be approximately 1.00056
        # for 889 rows, rather than exactly 1.
        if abs(std - 1.0) < 0.01:
            print("  Standard deviation is approximately 1: PASS")
        else:
            print("  Standard deviation is approximately 1: CHECK")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved EDA-standardized dataset to: {OUTPUT_PATH}")
    print("\nStage 2.6 complete.")


if __name__ == "__main__":
    main()
