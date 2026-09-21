"""Analytics 2.12 — Random Forest GridSearchCV and OOB evaluation."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV


BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "splits"
PREPROCESSING_DIR = BASE_DIR / "preprocessing"
OUTPUT_DIR = BASE_DIR / "results" / "stage_2_12"
PLOTS_DIR = BASE_DIR / "plots" / "stage_2_12"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
    X_test = pd.read_csv(SPLIT_DIR / "X_test.csv")
    y_train = pd.read_csv(SPLIT_DIR / "y_train.csv")["survived"]
    y_test = pd.read_csv(SPLIT_DIR / "y_test.csv")["survived"]

    preprocessor = joblib.load(PREPROCESSING_DIR / "preprocessor.joblib")
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.12: RANDOM FOREST GRIDSEARCHCV")
    print("=" * 80)
    print(f"Training rows: {X_train_t.shape[0]}")
    print(f"Training features: {X_train_t.shape[1]}")
    print(f"Test rows: {X_test_t.shape[0]}")

    rf = RandomForestClassifier(
        random_state=42,
        oob_score=True,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "max_features": ["sqrt", "log2", None],
    }

    combinations = 3 * 4 * 3
    print(f"Grid combinations: {combinations}")
    print("Cross-validation folds: 5")
    print("Scoring: F1")

    search = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train_t, y_train)

    best_rf = search.best_estimator_

    print("\n" + "=" * 80)
    print("BEST RANDOM FOREST")
    print("=" * 80)
    print("Best parameters:")
    print(search.best_params_)
    print(f"Best mean CV F1: {search.best_score_:.4f}")
    print(f"OOB score: {best_rf.oob_score_:.4f}")

    summary = pd.DataFrame([{
        "best_n_estimators": search.best_params_["n_estimators"],
        "best_max_depth": search.best_params_["max_depth"],
        "best_max_features": search.best_params_["max_features"],
        "best_cv_f1": search.best_score_,
        "oob_score": best_rf.oob_score_,
    }])
    summary.to_csv(OUTPUT_DIR / "gridsearch_summary.csv", index=False)

    cv = pd.DataFrame(search.cv_results_)
    cv[[
        "param_n_estimators", "param_max_depth", "param_max_features",
        "mean_test_score", "std_test_score", "mean_train_score",
        "rank_test_score"
    ]].sort_values("rank_test_score").to_csv(
        OUTPUT_DIR / "gridsearch_cv_results.csv", index=False
    )

    pred = best_rf.predict(X_test_t)
    proba = best_rf.predict_proba(X_test_t)[:, 1]
    metrics = pd.DataFrame([{
        "model": "Tuned Random Forest",
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }])
    metrics.to_csv(
        OUTPUT_DIR / "tuned_random_forest_test_metrics.csv", index=False
    )

    joblib.dump(best_rf, OUTPUT_DIR / "best_random_forest.joblib")

    top = cv.sort_values("rank_test_score").head(10).copy()
    labels = (
        "n=" + top["param_n_estimators"].astype(str)
        + ", depth=" + top["param_max_depth"].astype(str)
        + ", features=" + top["param_max_features"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(len(top)), top["mean_test_score"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean CV F1")
    ax.set_title("Top Random Forest GridSearchCV Configurations")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top_gridsearch_configurations.png", dpi=150)
    plt.close(fig)

    print("\nTest-set evaluation of selected estimator:")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- SAVED OUTPUTS ---")
    for p in [
        OUTPUT_DIR / "gridsearch_summary.csv",
        OUTPUT_DIR / "gridsearch_cv_results.csv",
        OUTPUT_DIR / "tuned_random_forest_test_metrics.csv",
        OUTPUT_DIR / "best_random_forest.joblib",
        PLOTS_DIR / "top_gridsearch_configurations.png",
    ]:
        print(p)
    print("\nStage 2.12 complete.")


if __name__ == "__main__":
    main()
