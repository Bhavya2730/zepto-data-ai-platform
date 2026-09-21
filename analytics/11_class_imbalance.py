"""
Analytics 2.11 — Class imbalance experiment.

Compares one classifier under three training strategies:
1. Baseline
2. class_weight="balanced"
3. SMOTE applied only to the training data

The same held-out test set is used for all three evaluations.
The Stage 2.8 preprocessor is fit on X_train only and is never refit
on the test set.

Classifier used: Logistic Regression
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "splits"
PREPROCESSING_DIR = BASE_DIR / "preprocessing"

OUTPUT_DIR = BASE_DIR / "results" / "stage_2_11"
PLOTS_DIR = BASE_DIR / "plots" / "stage_2_11"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_TEST_PATH = SPLIT_DIR / "X_test.csv"
Y_TRAIN_PATH = SPLIT_DIR / "y_train.csv"
Y_TEST_PATH = SPLIT_DIR / "y_test.csv"
PREPROCESSOR_PATH = PREPROCESSING_DIR / "preprocessor.joblib"


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "strategy": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }


def main() -> None:
    required_files = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
        PREPROCESSOR_PATH,
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Stage 2.7–2.8 files are missing:\n"
            + "\n".join(missing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["survived"]
    y_test = pd.read_csv(Y_TEST_PATH)["survived"]

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # The preprocessor was fitted during Stage 2.8 on X_train only.
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.11: CLASS IMBALANCE")
    print("=" * 80)

    print("\nOriginal training class distribution:")
    print(y_train.value_counts().sort_index())
    print("\nOriginal training class proportions:")
    print(
        y_train.value_counts(normalize=True)
        .sort_index()
        .rename("proportion")
    )

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    baseline = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    baseline.fit(X_train_transformed, y_train)

    baseline_result = evaluate_model(
        "Baseline",
        baseline,
        X_test_transformed,
        y_test,
    )

    # ------------------------------------------------------------------
    # 2. Class-weight balanced
    # ------------------------------------------------------------------
    balanced = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    balanced.fit(X_train_transformed, y_train)

    balanced_result = evaluate_model(
        "class_weight='balanced'",
        balanced,
        X_test_transformed,
        y_test,
    )

    # ------------------------------------------------------------------
    # 3. SMOTE
    # ------------------------------------------------------------------
    # SMOTE is applied ONLY to the already-transformed training data.
    # The held-out test set is never oversampled.
    smote = SMOTE(
        random_state=42,
    )
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_transformed,
        y_train,
    )

    print("\nSMOTE training class distribution:")
    print(
        pd.Series(y_train_smote)
        .value_counts()
        .sort_index()
    )

    smote_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    smote_model.fit(X_train_smote, y_train_smote)

    smote_result = evaluate_model(
        "SMOTE",
        smote_model,
        X_test_transformed,
        y_test,
    )

    results = pd.DataFrame(
        [
            baseline_result,
            balanced_result,
            smote_result,
        ]
    )

    # Keep confusion matrices in a separate CSV so the main comparison
    # table remains easy to use in later stages.
    confusion_rows = []
    for result in [
        baseline_result,
        balanced_result,
        smote_result,
    ]:
        cm = result["confusion_matrix"]
        confusion_rows.extend(
            [
                {
                    "strategy": result["strategy"],
                    "actual_class": 0,
                    "predicted_0": cm[0][0],
                    "predicted_1": cm[0][1],
                },
                {
                    "strategy": result["strategy"],
                    "actual_class": 1,
                    "predicted_0": cm[1][0],
                    "predicted_1": cm[1][1],
                },
            ]
        )

    comparison = results[
        [
            "strategy",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    ]

    comparison.to_csv(
        OUTPUT_DIR / "imbalance_comparison.csv",
        index=False,
    )

    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT_DIR / "confusion_matrices.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("IMBALANCE COMPARISON")
    print("=" * 80)
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nConfusion matrices:")
    for result in [
        baseline_result,
        balanced_result,
        smote_result,
    ]:
        print(f"\n{result['strategy']}")
        print(result["confusion_matrix"])

    # ------------------------------------------------------------------
    # Comparison chart: precision, recall, F1
    # ------------------------------------------------------------------
    metrics_to_plot = ["precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(9, 6))

    x = range(len(comparison))
    width = 0.25

    for offset, metric in zip(
        [-width, 0, width],
        metrics_to_plot,
    ):
        ax.bar(
            [value + offset for value in x],
            comparison[metric],
            width=width,
            label=metric.capitalize(),
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison["strategy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(
        "Class Imbalance Strategies — Precision, Recall and F1"
    )
    ax.legend()

    fig.tight_layout()
    chart_path = PLOTS_DIR / "imbalance_metric_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Required written interpretation
    # ------------------------------------------------------------------
    baseline_row = comparison.iloc[0]
    balanced_row = comparison.iloc[1]
    smote_row = comparison.iloc[2]

    interpretation = f"""# Stage 2.11 — Class Imbalance Interpretation

## Setup

Logistic Regression was used as the classifier for the imbalance experiment.
All three strategies used the same stratified train/test split from Stage 2.7
and the same Stage 2.8 preprocessing.

The three training strategies were:

1. Baseline Logistic Regression.
2. Logistic Regression with `class_weight="balanced"`.
3. Logistic Regression trained after applying SMOTE to the training data only.

The held-out test set was not oversampled.

## Class balance

The original training set contained:

- Class 0: {int(y_train.value_counts().get(0, 0))}
- Class 1: {int(y_train.value_counts().get(1, 0))}

After SMOTE, the training set contained:

- Class 0: {int(pd.Series(y_train_smote).value_counts().get(0, 0))}
- Class 1: {int(pd.Series(y_train_smote).value_counts().get(1, 0))}

## Results

The baseline produced precision {baseline_row["precision"]:.4f},
recall {baseline_row["recall"]:.4f}, and F1 {baseline_row["f1"]:.4f}.

The `class_weight="balanced"` strategy produced precision
{balanced_row["precision"]:.4f}, recall {balanced_row["recall"]:.4f}, and
F1 {balanced_row["f1"]:.4f}.

The SMOTE strategy produced precision {smote_row["precision"]:.4f},
recall {smote_row["recall"]:.4f}, and F1 {smote_row["f1"]:.4f}.

## Interpretation

The imbalance strategies change the precision/recall trade-off because they
change how the classifier treats the minority class during training.
`class_weight="balanced"` changes the training loss weighting without creating
new observations, while SMOTE creates synthetic minority-class training
examples. The test set remains unchanged, so the three strategies can be
compared on the same held-out observations.

For deployment selection, these results should be considered together with
the Stage 2.10 classifier evaluation and the later Random Forest
GridSearchCV results rather than selecting a model from this experiment
alone.
"""

    interpretation_path = OUTPUT_DIR / "interpretation.md"
    interpretation_path.write_text(
        interpretation,
        encoding="utf-8",
    )

    print("\n--- SAVED OUTPUTS ---")
    print(OUTPUT_DIR / "imbalance_comparison.csv")
    print(OUTPUT_DIR / "confusion_matrices.csv")
    print(interpretation_path)
    print(chart_path)

    print("\nStage 2.11 complete.")


if __name__ == "__main__":
    main()
