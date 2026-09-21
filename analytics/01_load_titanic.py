"""Analytics 2.1 — Load Titanic once and create the offline fallback."""

from pathlib import Path

import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"


def load_and_save_titanic() -> pd.DataFrame:
    # The single raw dataset load for the analytics module.
    df = sns.load_dataset("titanic")

    # Required offline fallback.
    df.to_csv(CSV_PATH, index=False)

    return df


def main() -> None:
    df = load_and_save_titanic()

    print("=" * 70)
    print("TITANIC DATASET — STAGE 2.1")
    print("=" * 70)

    print(f"\nSaved offline fallback to: {CSV_PATH}")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nStage 2.1 complete.")


if __name__ == "__main__":
    main()