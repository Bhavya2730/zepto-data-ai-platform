"""
Analytics 2.5 — Titanic multivariate data story.

This stage creates four distinct charts that combine multiple variables
and provides a concise written interpretation for each chart.

Input:
    analytics/titanic_cleaned_stage_2_2.csv

Outputs:
    analytics/plots/stage_2_5/*.png
    analytics/results/stage_2_5/*.csv

No Seaborn dataset loading occurs here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "titanic_cleaned_stage_2_2.csv"
PLOTS_DIR = BASE_DIR / "plots" / "stage_2_5"
RESULTS_DIR = BASE_DIR / "results" / "stage_2_5"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_PATH}\n"
            "Run Stage 2.2 first."
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.5: MULTIVARIATE DATA STORY")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Chart 1 — Survival by sex and passenger class
    # ------------------------------------------------------------------
    chart1 = (
        df.groupby(["pclass", "sex"], observed=False)["survived"]
        .mean()
        .reset_index()
    )
    chart1["survival_rate_percent"] = chart1["survived"] * 100

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=chart1,
        x="pclass",
        y="survival_rate_percent",
        hue="sex",
    )
    plt.title("Survival Rate by Passenger Class and Sex")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_survival_by_class_and_sex.png", dpi=150)
    plt.close()

    chart1.to_csv(
        RESULTS_DIR / "01_survival_by_class_and_sex.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Chart 2 — Age distribution by survival and sex
    # ------------------------------------------------------------------
    age_plot_df = df.dropna(subset=["age"]).copy()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=age_plot_df,
        x="sex",
        y="age",
        hue="survived",
    )
    plt.title("Age Distribution by Sex and Survival")
    plt.xlabel("Sex")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_age_by_sex_and_survival.png", dpi=150)
    plt.close()

    age_summary = (
        age_plot_df.groupby(["sex", "survived"], observed=False)["age"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    age_summary.to_csv(
        RESULTS_DIR / "02_age_by_sex_and_survival.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Chart 3 — Fare vs age, with survival and passenger class
    # ------------------------------------------------------------------
    scatter_df = df.dropna(subset=["age", "fare"]).copy()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=scatter_df,
        x="age",
        y="fare",
        hue="survived",
        style="pclass",
        size="pclass",
        sizes=(40, 140),
        alpha=0.7,
    )
    plt.title("Fare vs Age by Survival and Passenger Class")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_fare_vs_age_survival_class.png", dpi=150)
    plt.close()

    scatter_df[
        ["age", "fare", "survived", "pclass"]
    ].to_csv(
        RESULTS_DIR / "03_fare_vs_age_survival_class.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Chart 4 — Family size, passenger class, and survival
    # ------------------------------------------------------------------
    family_df = df.copy()
    family_df["family_size"] = (
        family_df["sibsp"] + family_df["parch"] + 1
    )

    family_summary = (
        family_df.groupby(
            ["pclass", "family_size"],
            observed=False,
        )["survived"]
        .mean()
        .reset_index()
    )
    family_summary["survival_rate_percent"] = (
        family_summary["survived"] * 100
    )

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=family_summary,
        x="family_size",
        y="survival_rate_percent",
        hue="pclass",
        marker="o",
    )
    plt.title("Survival Rate by Family Size and Passenger Class")
    plt.xlabel("Family Size")
    plt.ylabel("Survival Rate (%)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_survival_family_size_class.png", dpi=150)
    plt.close()

    family_summary.to_csv(
        RESULTS_DIR / "04_survival_family_size_class.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Written interpretations
    # ------------------------------------------------------------------
    interpretations = [
        (
            "Chart 1 — Survival Rate by Passenger Class and Sex",
            "Survival varies across both passenger class and sex. "
            "The grouped bars allow the effect of passenger class to be "
            "examined separately for females and males, showing that the "
            "relationship is not adequately described by either variable alone."
        ),
        (
            "Chart 2 — Age Distribution by Sex and Survival",
            "The box plots compare age distributions across sex and survival "
            "status. They show how the age profile of survivors and "
            "non-survivors differs within each sex, while also revealing "
            "differences in spread and potential extreme ages."
        ),
        (
            "Chart 3 — Fare vs Age by Survival and Passenger Class",
            "The scatter plot combines age, fare, survival, and passenger class. "
            "It shows how fare levels vary across ages and classes while "
            "allowing survival status to be compared within the same feature space."
        ),
        (
            "Chart 4 — Survival Rate by Family Size and Passenger Class",
            "Family size is derived from siblings/spouses plus parents/children "
            "plus the passenger. The lines show how survival changes with family "
            "size within each passenger class, providing a multivariate view of "
            "family structure, class, and survival."
        ),
    ]

    interpretation_path = RESULTS_DIR / "interpretations.md"
    interpretation_path.write_text(
        "# Stage 2.5 — Multivariate Chart Interpretations\n\n"
        + "\n\n".join(
            f"## {title}\n\n{interpretation}"
            for title, interpretation in interpretations
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n--- FOUR DISTINCT MULTIVARIATE CHARTS ---")
    for title, _ in interpretations:
        print(title)

    print("\n--- SAVED PLOTS ---")
    for path in sorted(PLOTS_DIR.glob("*.png")):
        print(path)

    print("\n--- WRITTEN INTERPRETATIONS ---")
    print(interpretation_path)

    print("\nStage 2.5 complete.")


if __name__ == "__main__":
    main()
