"""
Analytics 2.2 — Titanic dataset profiling and missing-value analysis.

This stage:
1. Reads the committed analytics/titanic.csv created in Stage 2.1.
2. Prints df.info(), df.describe(), and df.shape.
3. Calculates missing-value percentages for columns containing missing data.
4. Applies the assignment's missing-data thresholds:
   - <5%: drop affected rows
   - 5–30%: impute
   - >30%: decide whether to drop the column or encode missing as a category,
     with an explicit justification.
5. Saves the resulting cleaned DataFrame as titanic_cleaned_stage_2_2.csv.

No Seaborn dataset loading occurs here.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "titanic.csv"
OUTPUT_PATH = BASE_DIR / "titanic_cleaned_stage_2_2.csv"


def profile_dataset(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.2: PROFILE")
    print("=" * 80)

    print("\n--- df.info() ---")
    df.info()

    print("\n--- df.describe() ---")
    print(df.describe(include="all").to_string())

    print("\n--- df.shape ---")
    print(df.shape)

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_table = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percent": missing_pct,
        }
    )

    missing_table = missing_table[missing_table["missing_count"] > 0]

    print("\n--- Missing-value percentages ---")
    if missing_table.empty:
        print("No missing values found.")
    else:
        print(missing_table.to_string())


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the assignment thresholds.

    Titanic missingness:
      - age: approximately 19.87% -> numeric median imputation
      - embarked: approximately 0.22% -> drop affected rows
      - deck: approximately 77.10% -> retain as an explicit 'Unknown'
        category because deck is potentially informative and the assignment
        permits encoding missing values as a category when >30%.
    """
    cleaned = df.copy()

    print("\n" + "=" * 80)
    print("MISSING-VALUE HANDLING DECISIONS")
    print("=" * 80)

    missing = cleaned.isna().sum()
    missing_pct = missing / len(cleaned) * 100

    for column in cleaned.columns:
        if missing[column] == 0:
            continue

        pct = missing_pct[column]

        if pct < 5:
            print(
                f"{column}: {pct:.2f}% missing (<5%) -> "
                "drop rows containing missing values."
            )

            cleaned = cleaned.dropna(subset=[column])

        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                median_value = cleaned[column].median()
                cleaned[column] = cleaned[column].fillna(median_value)

                print(
                    f"{column}: {pct:.2f}% missing (5–30%) -> "
                    f"median imputation using {median_value:.2f}."
                )
            else:
                mode = cleaned[column].mode(dropna=True)
                if mode.empty:
                    raise ValueError(
                        f"Cannot determine a mode for column '{column}'."
                    )

                mode_value = mode.iloc[0]
                cleaned[column] = cleaned[column].fillna(mode_value)

                print(
                    f"{column}: {pct:.2f}% missing (5–30%) -> "
                    f"mode imputation using '{mode_value}'."
                )

        else:
            # The assignment permits either dropping the column or encoding
            # missing values as a category. Deck is retained as "Unknown".
            cleaned[column] = cleaned[column].astype("object").fillna("Unknown")

            print(
                f"{column}: {pct:.2f}% missing (>30%) -> "
                "retain column and encode missing values as 'Unknown'."
            )
            print(
                "Justification: deck has substantial missingness, but the "
                "missingness itself may carry information; retaining it as "
                "an explicit category avoids discarding the entire feature."
            )

    return cleaned


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_PATH}\n"
            "Run Stage 2.1 first."
        )

    df = pd.read_csv(INPUT_PATH)

    profile_dataset(df)

    cleaned_df = handle_missing_values(df)

    print("\n" + "=" * 80)
    print("POST-CLEANING CHECK")
    print("=" * 80)
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape:  {cleaned_df.shape}")

    remaining_missing = cleaned_df.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]

    print("\nRemaining missing values:")
    if remaining_missing.empty:
        print("None")
    else:
        print(remaining_missing.to_string())

    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned Stage 2.2 dataset to: {OUTPUT_PATH}")
    print("Stage 2.2 complete.")


if __name__ == "__main__":
    main()
