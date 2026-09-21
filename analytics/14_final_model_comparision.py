"""
Analytics 2.14 — Final model comparison and deployment recommendation.

Combines:
- Stage 2.10 baseline classifier evaluation
- Stage 2.11 imbalance experiment
- Stage 2.12 tuned Random Forest
- Stage 2.13 fare regression

The classifier and regression metrics are kept in distinct metric groups.
The final written recommendation concerns only the classification deployment
choice, as required by the assignment.
"""

from pathlib import Path

import pandas as pd


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_210 = BASE_DIR / "results" / "stage_2_10"
RESULTS_211 = BASE_DIR / "results" / "stage_2_11"
RESULTS_212 = BASE_DIR / "results" / "stage_2_12"
RESULTS_213 = BASE_DIR / "results" / "stage_2_13"

OUTPUT_DIR = BASE_DIR / "results" / "stage_2_14"


# ================================================================
# MAIN
# ================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # CHECK REQUIRED FILES
    # ============================================================

    required_files = [
        RESULTS_210 / "classifier_comparison.csv",
        RESULTS_211 / "imbalance_comparison.csv",
        RESULTS_212 / "gridsearch_summary.csv",
        RESULTS_212 / "tuned_random_forest_test_metrics.csv",
        RESULTS_213 / "fare_regression_metrics.csv",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required previous-stage results are missing:\n"
            + "\n".join(missing)
        )

    # ============================================================
    # LOAD CLASSIFIER RESULTS
    # ============================================================

    classifier = pd.read_csv(
        RESULTS_210 / "classifier_comparison.csv"
    )

    imbalance = pd.read_csv(
        RESULTS_211 / "imbalance_comparison.csv"
    )

    tuned_rf = pd.read_csv(
        RESULTS_212 / "tuned_random_forest_test_metrics.csv"
    ).iloc[0]

    grid_summary = pd.read_csv(
        RESULTS_212 / "gridsearch_summary.csv"
    ).iloc[0]

    # FIX: Convert NaN max_features back to "None"
    best_max_features = grid_summary["best_max_features"]

    if pd.isna(best_max_features):
        best_max_features = "None"

    # ============================================================
    # CLASSIFIER COMPARISON
    # ============================================================

    classifier_table = classifier.copy()

    classifier_table["evaluation_source"] = (
        "Stage 2.10"
    )

    # Add tuned Random Forest as another classifier row
    tuned_row = pd.DataFrame(
        [
            {
                "model": "Tuned Random Forest",
                "accuracy": tuned_rf["accuracy"],
                "precision": tuned_rf["precision"],
                "recall": tuned_rf["recall"],
                "f1": tuned_rf["f1"],
                "roc_auc": tuned_rf["roc_auc"],
                "evaluation_source": "Stage 2.12",
            }
        ]
    )

    classifier_table = pd.concat(
        [
            classifier_table,
            tuned_row,
        ],
        ignore_index=True,
    )

    classifier_table.to_csv(
        OUTPUT_DIR / "classifier_model_comparison.csv",
        index=False,
    )

    # ============================================================
    # IMBALANCE COMPARISON
    # ============================================================

    imbalance.to_csv(
        OUTPUT_DIR / "imbalance_strategy_comparison.csv",
        index=False,
    )

    # ============================================================
    # REGRESSION RESULTS
    # ============================================================

    regression = pd.read_csv(
        RESULTS_213 / "fare_regression_metrics.csv"
    )

    regression_table = regression[
        [
            "model",
            "target",
            "MAE",
            "RMSE",
            "R2",
            "Adjusted_R2",
        ]
    ].copy()

    regression_table.to_csv(
        OUTPUT_DIR / "regression_model_comparison.csv",
        index=False,
    )

    # ============================================================
    # GRIDSEARCH SUMMARY
    # ============================================================

    gridsearch_table = pd.DataFrame(
        [
            {
                "best_n_estimators":
                    grid_summary["best_n_estimators"],

                "best_max_depth":
                    grid_summary["best_max_depth"],

                "best_max_features":
                    best_max_features,

                "best_cv_f1":
                    grid_summary["best_cv_f1"],

                "oob_score":
                    grid_summary["oob_score"],
            }
        ]
    )

    gridsearch_table.to_csv(
        OUTPUT_DIR / "random_forest_gridsearch_summary.csv",
        index=False,
    )

    # ============================================================
    # GET CLASSIFIER METRICS
    # ============================================================

    logistic = classifier.loc[
        classifier["model"] == "Logistic Regression"
    ].iloc[0]

    decision_tree = classifier.loc[
        classifier["model"] == "Decision Tree"
    ].iloc[0]

    random_forest = classifier.loc[
        classifier["model"] == "Random Forest"
    ].iloc[0]

    # ============================================================
    # DEPLOYMENT RECOMMENDATION
    # ============================================================

    recommendation = f"""
# Stage 2.14 — Final Model Comparison and Deployment Recommendation

## Classification model comparison

The classification models were evaluated on the same held-out Titanic test
set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | {logistic["accuracy"]:.4f} | {logistic["precision"]:.4f} | {logistic["recall"]:.4f} | {logistic["f1"]:.4f} | {logistic["roc_auc"]:.4f} |
| Decision Tree | {decision_tree["accuracy"]:.4f} | {decision_tree["precision"]:.4f} | {decision_tree["recall"]:.4f} | {decision_tree["f1"]:.4f} | {decision_tree["roc_auc"]:.4f} |
| Random Forest | {random_forest["accuracy"]:.4f} | {random_forest["precision"]:.4f} | {random_forest["recall"]:.4f} | {random_forest["f1"]:.4f} | {random_forest["roc_auc"]:.4f} |
| Tuned Random Forest | {tuned_rf["accuracy"]:.4f} | {tuned_rf["precision"]:.4f} | {tuned_rf["recall"]:.4f} | {tuned_rf["f1"]:.4f} | {tuned_rf["roc_auc"]:.4f} |

## Imbalance experiment

The Logistic Regression imbalance experiment produced:

| Strategy | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline | {imbalance.loc[imbalance["strategy"] == "Baseline", "accuracy"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"] == "Baseline", "precision"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"] == "Baseline", "recall"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"] == "Baseline", "f1"].iloc[0]:.4f} |
| class_weight="balanced" | {imbalance.loc[imbalance["strategy"].str.contains("class_weight"), "accuracy"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"].str.contains("class_weight"), "precision"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"].str.contains("class_weight"), "recall"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"].str.contains("class_weight"), "f1"].iloc[0]:.4f} |
| SMOTE | {imbalance.loc[imbalance["strategy"] == "SMOTE", "accuracy"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"] == "SMOTE", "precision"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"] == "SMOTE", "recall"].iloc[0]:.4f} | {imbalance.loc[imbalance["strategy"] == "SMOTE", "f1"].iloc[0]:.4f} |

The imbalance experiment increased recall from 0.7059 for the baseline
Logistic Regression to 0.7794 for both balanced weighting and SMOTE.
However, precision and F1 decreased relative to the baseline. SMOTE retained
higher precision and F1 than the class-weighted Logistic Regression while
achieving the same recall.

## Random Forest tuning

The selected Random Forest configuration was:

- `n_estimators`: {grid_summary["best_n_estimators"]}
- `max_depth`: {grid_summary["best_max_depth"]}
- `max_features`: {best_max_features}
- Best mean CV F1: {grid_summary["best_cv_f1"]:.4f}
- OOB score: {grid_summary["oob_score"]:.4f}

On the untouched test set, the tuned Random Forest produced accuracy
{tuned_rf["accuracy"]:.4f}, precision {tuned_rf["precision"]:.4f},
recall {tuned_rf["recall"]:.4f}, F1 {tuned_rf["f1"]:.4f}, and ROC-AUC
{tuned_rf["roc_auc"]:.4f}.

## Regression results

Regression metrics are kept separate from classification metrics because
they measure a different target and task.

| Model | Target | MAE | RMSE | R² | Adjusted R² |
|---|---|---:|---:|---:|---:|
| {regression.iloc[0]["model"]} | {regression.iloc[0]["target"]} | {regression.iloc[0]["MAE"]:.4f} | {regression.iloc[0]["RMSE"]:.4f} | {regression.iloc[0]["R2"]:.4f} | {regression.iloc[0]["Adjusted_R2"]:.4f} |

The regression residual analysis found a residual spread that changes
substantially across predicted fare values, which is consistent with
heteroscedasticity.

## Final classifier deployment recommendation

The **Tuned Random Forest** is recommended for deployment for the
classification task. It achieved the highest F1 among the evaluated
classifier configurations at **{tuned_rf["f1"]:.4f}**, compared with
**{random_forest["f1"]:.4f}** for the baseline Random Forest,
**{logistic["f1"]:.4f}** for Logistic Regression, and
**{decision_tree["f1"]:.4f}** for the Decision Tree. Its recall was
**{tuned_rf["recall"]:.4f}** and accuracy was **{tuned_rf["accuracy"]:.4f}**,
while its ROC-AUC was **{tuned_rf["roc_auc"]:.4f}**. Logistic Regression
had higher precision (**{logistic["precision"]:.4f}**) and ROC-AUC
(**{logistic["roc_auc"]:.4f}**) than the tuned Random Forest, so the
comparison should consider the complete set of observed classification
metrics rather than a single metric.
"""

    recommendation_path = (
        OUTPUT_DIR / "final_model_comparison.md"
    )

    recommendation_path.write_text(
        recommendation,
        encoding="utf-8",
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.14: FINAL MODEL COMPARISON")
    print("=" * 80)

    print("\nCLASSIFIER COMPARISON")

    print(
        classifier_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nIMBALANCE COMPARISON")

    print(
        imbalance.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nRANDOM FOREST GRIDSEARCH SUMMARY")

    print(
        gridsearch_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nREGRESSION COMPARISON")

    print(
        regression_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nFINAL DEPLOYMENT RECOMMENDATION")

    print("Tuned Random Forest")
    print(
        "F1:",
        f"{tuned_rf['f1']:.4f}"
    )

    print(
        "Accuracy:",
        f"{tuned_rf['accuracy']:.4f}"
    )

    print(
        "Recall:",
        f"{tuned_rf['recall']:.4f}"
    )

    print(
        "Precision:",
        f"{tuned_rf['precision']:.4f}"
    )

    print(
        "ROC-AUC:",
        f"{tuned_rf['roc_auc']:.4f}"
    )

    print("\n--- SAVED OUTPUTS ---")

    print(
        OUTPUT_DIR
        / "classifier_model_comparison.csv"
    )

    print(
        OUTPUT_DIR
        / "imbalance_strategy_comparison.csv"
    )

    print(
        OUTPUT_DIR
        / "random_forest_gridsearch_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "regression_model_comparison.csv"
    )

    print(
        recommendation_path
    )

    print("\nStage 2.14 complete.")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()