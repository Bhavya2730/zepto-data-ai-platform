"""
Analytics 2.9 — Train three classifiers on the same train/test split.

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

All three use the SAME training-only preprocessing from Stage 2.8.
The Decision Tree is also rendered with feature names and class names.

No test data is used to fit preprocessing or models.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "splits"
PREPROCESSING_DIR = BASE_DIR / "preprocessing"
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots" / "stage_2_9"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_TEST_PATH = SPLIT_DIR / "X_test.csv"
Y_TRAIN_PATH = SPLIT_DIR / "y_train.csv"
Y_TEST_PATH = SPLIT_DIR / "y_test.csv"
PREPROCESSOR_PATH = PREPROCESSING_DIR / "preprocessor.joblib"

RANDOM_STATE = 42


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
            "Required Stage 2.7/2.8 files are missing:\n"
            + "\n".join(missing)
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH)["survived"]
    y_test = pd.read_csv(Y_TEST_PATH)["survived"]

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Use the exact Stage 2.8 fitted preprocessor.
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # sklearn may return feature names as a NumPy array.
    # Convert them to a Python list because plot_tree requires a list.
    feature_names = preprocessor.get_feature_names_out().tolist()

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.9: THREE CLASSIFIERS")
    print("=" * 80)

    print("\n--- INPUT SHAPES ---")
    print(f"X_train transformed: {X_train_transformed.shape}")
    print(f"X_test transformed:  {X_test_transformed.shape}")
    print(f"y_train:              {y_train.shape}")
    print(f"y_test:               {y_test.shape}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=5,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }

    trained_models = {}

    print("\n--- MODEL TRAINING ---")

    for name, model in models.items():
        model.fit(X_train_transformed, y_train)
        trained_models[name] = model

        safe_name = name.lower().replace(" ", "_")

        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib",
        )

        print(f"{name}: trained successfully")

    # ------------------------------------------------------------------
    # Decision Tree visualization
    # ------------------------------------------------------------------

    decision_tree = trained_models["Decision Tree"]

    plt.figure(figsize=(24, 12))

    plot_tree(
        decision_tree,
        feature_names=feature_names,
        class_names=["Did not survive", "Survived"],
        filled=True,
        rounded=True,
        fontsize=7,
    )

    plt.title("Decision Tree Classifier")
    plt.tight_layout()

    tree_path = PLOTS_DIR / "decision_tree.png"
    plt.savefig(tree_path, dpi=180, bbox_inches="tight")
    plt.close()

    print("\n--- DECISION TREE ---")
    print(f"Feature count: {len(feature_names)}")
    print(f"Tree depth: {decision_tree.get_depth()}")
    print(f"Leaf count: {decision_tree.get_n_leaves()}")
    print(f"Saved visualization: {tree_path}")

    # ------------------------------------------------------------------
    # Basic prediction-count sanity check.
    # Full evaluation belongs to Stage 2.10.
    # ------------------------------------------------------------------

    print("\n--- PREDICTION SANITY CHECK ---")

    for name, model in trained_models.items():
        predictions = model.predict(X_test_transformed)
        prediction_counts = (
            pd.Series(predictions)
            .value_counts()
            .sort_index()
        )

        print(f"\n{name}")
        print(
            "Predicted class counts:\n"
            f"{prediction_counts.to_string()}"
        )

    print("\n--- SAVED MODELS ---")

    for path in sorted(MODEL_DIR.glob("*.joblib")):
        print(path)

    print("\nStage 2.9 complete.")


if __name__ == "__main__":
    main()