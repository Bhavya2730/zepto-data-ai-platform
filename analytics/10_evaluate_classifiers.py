"""
Analytics 2.10 — Full classifier evaluation.

Evaluates:
- Logistic Regression
- Decision Tree
- Random Forest

Metrics:
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1
- ROC curve
- ROC-AUC

All models use the same Stage 2.7 split and Stage 2.8 preprocessing.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "splits"
PREPROCESSING_DIR = BASE_DIR / "preprocessing"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "results" / "stage_2_10"
PLOTS_DIR = BASE_DIR / "plots" / "stage_2_10"

X_TEST_PATH = SPLIT_DIR / "X_test.csv"
Y_TEST_PATH = SPLIT_DIR / "y_test.csv"
PREPROCESSOR_PATH = PREPROCESSING_DIR / "preprocessor.joblib"


def main() -> None:
    required_files = [
        X_TEST_PATH,
        Y_TEST_PATH,
        PREPROCESSOR_PATH,
        MODEL_DIR / "logistic_regression.joblib",
        MODEL_DIR / "decision_tree.joblib",
        MODEL_DIR / "random_forest.joblib",
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Required Stage 2.7–2.9 files are missing:\n"
            + "\n".join(missing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)["survived"]

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_test_transformed = preprocessor.transform(X_test)

    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.joblib"
        ),
        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree.joblib"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest.joblib"
        ),
    }

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.10: CLASSIFIER EVALUATION")
    print("=" * 80)

    metrics_rows = []
    roc_data = {}

    for name, model in models.items():
        predictions = model.predict(X_test_transformed)
        probabilities = model.predict_proba(X_test_transformed)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
        auc = roc_auc_score(y_test, probabilities)

        cm = confusion_matrix(y_test, predictions)

        metrics_rows.append(
            {
                "model": name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
            }
        )

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_data[name] = (fpr, tpr, auc)

        # Save confusion matrix data.
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        safe_name = name.lower().replace(" ", "_")
        cm_df.to_csv(
            OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv"
        )

        # Save confusion matrix visualization.
        fig, ax = plt.subplots(figsize=(6, 5))
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Did not survive", "Survived"],
        )
        display.plot(ax=ax)
        ax.set_title(f"{name} — Confusion Matrix")
        fig.tight_layout()
        fig.savefig(
            PLOTS_DIR / f"{safe_name}_confusion_matrix.png",
            dpi=150,
        )
        plt.close(fig)

        print(f"\n--- {name.upper()} ---")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("Confusion matrix:")
        print(cm)

    comparison = pd.DataFrame(metrics_rows)

    # Keep the table in a stable model order rather than ranking models.
    model_order = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    ]
    comparison["model"] = pd.Categorical(
        comparison["model"],
        categories=model_order,
        ordered=True,
    )
    comparison = comparison.sort_values("model").reset_index(drop=True)

    comparison.to_csv(
        OUTPUT_DIR / "classifier_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("CLASSIFIER COMPARISON TABLE")
    print("=" * 80)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Single ROC comparison plot.
    fig, ax = plt.subplots(figsize=(9, 7))

    for name in model_order:
        fpr, tpr, auc = roc_data[name]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_title("ROC Curves — Titanic Classifiers")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()

    roc_path = PLOTS_DIR / "roc_curves.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    print("\n--- SAVED OUTPUTS ---")
    print(OUTPUT_DIR / "classifier_comparison.csv")
    print(roc_path)

    for path in sorted(OUTPUT_DIR.glob("*_confusion_matrix.csv")):
        print(path)

    print("\nStage 2.10 complete.")


if __name__ == "__main__":
    main()
