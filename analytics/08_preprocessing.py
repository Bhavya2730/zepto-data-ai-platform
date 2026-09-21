"""
Analytics 2.8 — Training-only preprocessing pipeline.

This stage reads the train/test split created in Stage 2.7 and builds
a ColumnTransformer + Pipeline.

Requirements covered:
- Numeric preprocessing: median imputation + StandardScaler
- Categorical preprocessing: most-frequent imputation + OneHotEncoder
- Preprocessing is fitted on X_train only
- X_test is transformed using the already-fitted preprocessing
- No estimator is trained in this stage

The fitted preprocessor is saved for inspection/reuse in later stages.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "splits"
OUTPUT_DIR = BASE_DIR / "preprocessing"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_TEST_PATH = SPLIT_DIR / "X_test.csv"

PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.joblib"
TRAIN_TRANSFORMED_PATH = OUTPUT_DIR / "X_train_transformed.csv"
TEST_TRANSFORMED_PATH = OUTPUT_DIR / "X_test_transformed.csv"


# Keep modeling features focused on variables that are available as
# explanatory inputs. The target "survived" was already removed in Stage 2.7.
#
# The original Titanic dataset also contains derived/duplicate variables
# such as "alive", "class", "who", "adult_male", and "alone".
# To avoid target leakage or redundant representations, this stage uses:
#   Numeric: pclass, age, sibsp, parch, fare
#   Categorical: sex, embarked
#
# "deck" is retained as a categorical variable because Stage 2.2 explicitly
# handled its missingness as an "Unknown" category.
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


def build_preprocessor() -> ColumnTransformer:
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


def main() -> None:
    if not X_TRAIN_PATH.exists() or not X_TEST_PATH.exists():
        raise FileNotFoundError(
            "Stage 2.7 split files were not found. "
            "Run analytics/07_train_test_split.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    required_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_features = [
        column
        for column in required_features
        if column not in X_train.columns
    ]

    if missing_features:
        raise ValueError(
            f"Required modeling features are missing: {missing_features}"
        )

    print("=" * 80)
    print("TITANIC DATASET — STAGE 2.8: TRAINING-ONLY PREPROCESSING")
    print("=" * 80)

    print("\n--- FEATURES ---")
    print(f"Numeric: {NUMERIC_FEATURES}")
    print(f"Categorical: {CATEGORICAL_FEATURES}")

    print("\n--- PREPROCESSOR ---")
    preprocessor = build_preprocessor()

    # CRITICAL: fit only on training data.
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Test data is transformed using the training-fitted preprocessor.
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_transformed_df = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_transformed_df = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )

    print("\n--- SHAPES ---")
    print(f"Original X_train shape:     {X_train.shape}")
    print(f"Transformed X_train shape:  {X_train_transformed_df.shape}")
    print(f"Original X_test shape:      {X_test.shape}")
    print(f"Transformed X_test shape:   {X_test_transformed_df.shape}")

    print("\n--- TRANSFORMED FEATURE NAMES ---")
    for name in feature_names:
        print(name)

    print("\n--- TRAINING-ONLY FIT CHECK ---")
    print(
        "Preprocessor fit: X_train only"
    )
    print(
        "X_test operation: transform only (no fitting)"
    )

    # Save transformed matrices as diagnostic artifacts for this stage.
    X_train_transformed_df.to_csv(
        TRAIN_TRANSFORMED_PATH,
        index=False,
    )
    X_test_transformed_df.to_csv(
        TEST_TRANSFORMED_PATH,
        index=False,
    )

    # Save the fitted preprocessor so later modeling code can use the exact
    # same transformation without refitting on test/full data.
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("\n--- SAVED OUTPUTS ---")
    print(TRAIN_TRANSFORMED_PATH)
    print(TEST_TRANSFORMED_PATH)
    print(PREPROCESSOR_PATH)

    print("\nStage 2.8 complete.")


if __name__ == "__main__":
    main()
