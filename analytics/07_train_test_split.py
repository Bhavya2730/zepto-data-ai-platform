"""
Analytics 2.7 — Stratified train/test split.

This stage:
- Reads the cleaned Titanic dataset from Stage 2.2.
- Separates target `survived` from the features.
- Performs a stratified train/test split.
- Reports class proportions before and after splitting.
- Saves the split datasets for the next modeling stages.

Important:
No imputation, encoding, scaling, SMOTE, or model fitting happens here.
Those operations belong after the split and must be fitted using training data only.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "titanic_cleaned_stage_2_2.csv"
SPLIT_DIR = BASE_DIR / "splits"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def class_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    percentages = y.value_counts(normalize=True).sort_index() * 100

    return pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_PATH}\n"
            "Run Stage 2.2 first."
        )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    if "survived" not in df.columns:
        raise ValueError("Target column 'survived' was not found.")

    X = df.drop(columns=["survived"])
    y = df["survived"]

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.7: STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 80)

    print("\n--- TARGET ---")
    print("Target variable: survived")
    print("Target classes: 0 = did not survive, 1 = survived")

    print("\n--- FULL DATASET CLASS BALANCE ---")
    print(class_distribution(y).to_string())

    # Stratification preserves approximately the same target-class
    # proportions in train and test as in the complete cleaned dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n--- SPLIT SIZES ---")
    print(f"Full dataset: {len(df)} rows")
    print(f"Training set: {len(X_train)} rows")
    print(f"Test set:     {len(X_test)} rows")
    print(f"Test size:    {TEST_SIZE:.0%}")
    print(f"Random state: {RANDOM_STATE}")

    print("\n--- TRAINING CLASS BALANCE ---")
    print(class_distribution(y_train).to_string())

    print("\n--- TEST CLASS BALANCE ---")
    print(class_distribution(y_test).to_string())

    # Compare percentages explicitly to demonstrate that stratification
    # preserved the target distribution.
    comparison = pd.DataFrame(
        {
            "full_percent": y.value_counts(normalize=True).sort_index() * 100,
            "train_percent": (
                y_train.value_counts(normalize=True).sort_index() * 100
            ),
            "test_percent": (
                y_test.value_counts(normalize=True).sort_index() * 100
            ),
        }
    ).round(2)

    print("\n--- CLASS-PROPORTION COMPARISON ---")
    print(comparison.to_string())

    comparison.to_csv(
        SPLIT_DIR / "class_distribution_comparison.csv"
    )

    # Save split data. These files contain no fitted preprocessing objects.
    X_train.to_csv(SPLIT_DIR / "X_train.csv", index=False)
    X_test.to_csv(SPLIT_DIR / "X_test.csv", index=False)
    y_train.to_csv(SPLIT_DIR / "y_train.csv", index=False)
    y_test.to_csv(SPLIT_DIR / "y_test.csv", index=False)

    print("\n--- SAVED SPLITS ---")
    print(SPLIT_DIR / "X_train.csv")
    print(SPLIT_DIR / "X_test.csv")
    print(SPLIT_DIR / "y_train.csv")
    print(SPLIT_DIR / "y_test.csv")

    print("\n--- WHY STRATIFY? ---")
    print(
        "The target is a binary survival outcome. Stratification keeps the "
        "proportion of survivors and non-survivors approximately consistent "
        "between the training and test sets, making the evaluation split "
        "representative of the original target distribution."
    )

    print("\nStage 2.7 complete.")


if __name__ == "__main__":
    main()
