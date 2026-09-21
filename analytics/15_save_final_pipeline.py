"""
Analytics 2.15 — Persist and reload the final classification pipeline.

The selected classifier from Stage 2.14 is the tuned Random Forest.
This stage builds one complete sklearn Pipeline containing:

1. Raw-input preprocessing
2. Tuned Random Forest estimator

The pipeline is fitted on the training data only, saved with joblib,
reloaded, and then used to predict from a raw passenger record.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "splits"
OUTPUT_DIR = BASE_DIR / "results" / "stage_2_15"
MODEL_DIR = BASE_DIR / "models"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
Y_TRAIN_PATH = SPLIT_DIR / "y_train.csv"
X_TEST_PATH = SPLIT_DIR / "X_test.csv"
Y_TEST_PATH = SPLIT_DIR / "y_test.csv"


# These are the same raw predictor columns used in Stage 2.8.
NUMERIC_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

CATEGORICAL_FEATURES = [
    "sex",
    "embarked",
    "deck",
]


def build_preprocessor():
    """Create the raw-input preprocessing component."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_final_pipeline():
    """Create the complete preprocessing + tuned Random Forest pipeline."""
    preprocessor = build_preprocessor()

    tuned_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        max_features=None,
        random_state=42,
        oob_score=True,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", tuned_rf),
        ]
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Stage 2.7 split files are missing:\n"
            + "\n".join(missing)
        )

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["survived"]

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)["survived"]

    # Ensure the raw data contains exactly the expected predictor columns.
    X_train = X_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_test = X_test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.15: FINAL PIPELINE PERSISTENCE")
    print("=" * 80)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print("Raw predictor columns:")
    print(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    pipeline = build_final_pipeline()

    print("\nFITTING COMPLETE RAW-INPUT PIPELINE")
    pipeline.fit(X_train, y_train)

    classifier = pipeline.named_steps["classifier"]

    print("Pipeline fitted successfully.")
    print(f"OOB score: {classifier.oob_score_:.4f}")

    # Save one complete pipeline: preprocessing + final estimator.
    model_path = MODEL_DIR / "final_tuned_random_forest_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    print(f"\nSaved complete pipeline:")
    print(model_path)

    # Reload the exact persisted pipeline.
    reloaded_pipeline = joblib.load(model_path)

    print("\nPIPELINE RELOAD TEST")
    print("Pipeline reloaded successfully.")

    # Predict on raw test rows to verify that preprocessing is embedded.
    test_predictions = reloaded_pipeline.predict(X_test)
    test_probabilities = reloaded_pipeline.predict_proba(X_test)[:, 1]

    test_output = X_test.copy()
    test_output["actual_survived"] = y_test.to_numpy()
    test_output["predicted_survived"] = test_predictions
    test_output["survival_probability"] = test_probabilities

    test_output_path = OUTPUT_DIR / "reloaded_pipeline_test_predictions.csv"
    test_output.to_csv(test_output_path, index=False)

    # Raw-input example. This deliberately contains a missing age to prove
    # the persisted pipeline handles preprocessing at prediction time.
    raw_passenger = pd.DataFrame(
        [
            {
                "pclass": 3,
                "age": None,
                "sibsp": 0,
                "parch": 0,
                "fare": 8.05,
                "sex": "male",
                "embarked": "S",
                "deck": "Unknown",
            }
        ]
    )

    raw_prediction = reloaded_pipeline.predict(raw_passenger)[0]
    raw_probability = reloaded_pipeline.predict_proba(
        raw_passenger
    )[0, 1]

    print("\nRAW-INPUT PREDICTION TEST")
    print("Input:")
    print(raw_passenger.to_string(index=False))
    print(f"Predicted survived class: {raw_prediction}")
    print(f"Predicted survival probability: {raw_probability:.4f}")

    raw_output = raw_passenger.copy()
    raw_output["predicted_survived"] = raw_prediction
    raw_output["survival_probability"] = raw_probability

    raw_output_path = OUTPUT_DIR / "raw_input_prediction.csv"
    raw_output.to_csv(raw_output_path, index=False)

    # Save a compact pipeline specification for reproducibility.
    specification = f"""# Stage 2.15 — Final Pipeline Specification

## Final model

Tuned Random Forest classifier.

Parameters:

- `n_estimators = 200`
- `max_depth = 10`
- `max_features = None`
- `random_state = 42`
- `oob_score = True`

OOB score after fitting the training data: **{classifier.oob_score_:.4f}**

## Complete pipeline

The persisted joblib object contains both:

1. Raw-input preprocessing
2. Tuned Random Forest classifier

### Numeric preprocessing

- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`

Median imputation followed by StandardScaler.

### Categorical preprocessing

- `sex`
- `embarked`
- `deck`

Most-frequent imputation followed by one-hot encoding with
`handle_unknown="ignore"`.

## Persistence test

The complete pipeline was saved and successfully reloaded with `joblib`.

It was then used directly on raw, untransformed test rows and on a raw
passenger record containing a missing `age` value.

This confirms that preprocessing is embedded in the persisted pipeline rather
than requiring separate manual transformation at inference time.
"""

    specification_path = OUTPUT_DIR / "pipeline_specification.md"
    specification_path.write_text(
        specification,
        encoding="utf-8",
    )

    print("\n--- SAVED OUTPUTS ---")
    print(model_path)
    print(test_output_path)
    print(raw_output_path)
    print(specification_path)

    print("\nStage 2.15 complete.")


if __name__ == "__main__":
    main()
