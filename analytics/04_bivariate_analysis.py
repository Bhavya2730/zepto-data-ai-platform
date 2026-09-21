"""Analytics 2.4 — Titanic bivariate analysis and correlation."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "titanic_cleaned_stage_2_2.csv"
OUTPUT_DIR = BASE_DIR / "plots" / "stage_2_4"
RESULTS_DIR = BASE_DIR / "results" / "stage_2_4"
CORRELATION_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]

def survival_rate(series: pd.Series) -> float:
    return series.mean() * 100

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input file not found: {INPUT_PATH}\nRun Stage 2.2 first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.4: BIVARIATE ANALYSIS")
    print("=" * 80)

    print("\n--- SURVIVAL RATE BY SEX ---")
    sex_rates = (df.groupby("sex", observed=False)["survived"].mean().mul(100).round(2).sort_values(ascending=False))
    print(sex_rates.to_string())
    male_mask = df["sex"].eq("male")
    female_mask = df["sex"].eq("female")
    print(f"\nMale survival rate: {survival_rate(df.loc[male_mask, 'survived']):.2f}%")
    print(f"Female survival rate: {survival_rate(df.loc[female_mask, 'survived']):.2f}%")

    print("\n--- SURVIVAL RATE BY PCLASS ---")
    pclass_rates = df.groupby("pclass", observed=False)["survived"].mean().mul(100).round(2)
    print(pclass_rates.to_string())
    for pclass in sorted(df["pclass"].dropna().unique()):
        mask = df["pclass"].eq(pclass)
        print(f"Class {int(pclass)} survival rate: {survival_rate(df.loc[mask, 'survived']):.2f}%")

    print("\n--- SURVIVAL RATE BY SEX + PCLASS ---")
    combinations = []
    for sex in ["female", "male"]:
        for pclass in sorted(df["pclass"].dropna().unique()):
            mask = df["sex"].eq(sex) & df["pclass"].eq(pclass)
            count = int(mask.sum())
            rate = survival_rate(df.loc[mask, "survived"]) if count else float("nan")
            combinations.append({"sex": sex, "pclass": int(pclass), "passengers": count, "survival_rate_percent": round(rate, 2)})
    sex_pclass_rates = pd.DataFrame(combinations)
    print(sex_pclass_rates.to_string(index=False))
    non_first_class_mask = df["pclass"].eq(2) | df["pclass"].eq(3)
    print(f"\nClasses 2 OR 3 combined survival rate: {survival_rate(df.loc[non_first_class_mask, 'survived']):.2f}%")

    sex_rates.rename("survival_rate_percent").to_csv(RESULTS_DIR / "survival_by_sex.csv")
    pclass_rates.rename("survival_rate_percent").to_csv(RESULTS_DIR / "survival_by_pclass.csv")
    sex_pclass_rates.to_csv(RESULTS_DIR / "survival_by_sex_and_pclass.csv", index=False)

    print("\n" + "=" * 80)
    print("REQUIRED SIX-COLUMN CORRELATION MATRIX")
    print("=" * 80)
    correlation_matrix = df[CORRELATION_COLUMNS].corr()
    print("\nColumns included:")
    print(CORRELATION_COLUMNS)
    print("\nCorrelation matrix:")
    print(correlation_matrix.round(4).to_string())
    correlation_matrix.to_csv(RESULTS_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Titanic Correlation Matrix")
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    pairs = []
    for i, col1 in enumerate(CORRELATION_COLUMNS):
        for j in range(i + 1, len(CORRELATION_COLUMNS)):
            col2 = CORRELATION_COLUMNS[j]
            coefficient = correlation_matrix.loc[col1, col2]
            pairs.append({"feature_1": col1, "feature_2": col2, "correlation": coefficient, "absolute_correlation": abs(coefficient)})
    pairs_df = pd.DataFrame(pairs).sort_values("absolute_correlation", ascending=False).reset_index(drop=True)
    strongest_two = pairs_df.head(2)
    print("\n--- TWO STRONGEST ABSOLUTE OFF-DIAGONAL CORRELATIONS ---")
    print(strongest_two.to_string(index=False))
    pairs_df.to_csv(RESULTS_DIR / "all_correlation_pairs.csv", index=False)
    strongest_two.to_csv(RESULTS_DIR / "strongest_two_correlations.csv", index=False)

    print("\n--- SAVED OUTPUTS ---")
    print(heatmap_path)
    print(RESULTS_DIR / "survival_by_sex.csv")
    print(RESULTS_DIR / "survival_by_pclass.csv")
    print(RESULTS_DIR / "survival_by_sex_and_pclass.csv")
    print(RESULTS_DIR / "correlation_matrix.csv")
    print(RESULTS_DIR / "strongest_two_correlations.csv")
    print("\nStage 2.4 complete.")

if __name__ == "__main__":
    main()
