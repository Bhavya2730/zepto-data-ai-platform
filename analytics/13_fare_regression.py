"""
Analytics 2.13 — Titanic Fare Regression

Predict fare from other available Titanic passenger features.

Requirements:
- Multivariate Linear Regression
- Same train/test partition as Stage 2.7
- Preprocessing fitted only on training data
- MAE
- RMSE
- R²
- Adjusted R²
- Residual plot
- Actual vs predicted plot
- Heteroscedasticity interpretation
- Complete fitted pipeline saved and reloaded
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

CLEANED_PATH = BASE_DIR / "titanic_cleaned_stage_2_2.csv"

OUTPUT_DIR = BASE_DIR / "results" / "stage_2_13"
PLOTS_DIR = BASE_DIR / "plots" / "stage_2_13"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# ADJUSTED R-SQUARED
# ================================================================

def adjusted_r2(r2, n, p):
    """
    Calculate Adjusted R².
    """

    if n <= p + 1:
        return np.nan

    return 1 - (
        (1 - r2) * (n - 1) / (n - p - 1)
    )


# ================================================================
# MAIN
# ================================================================

def main():

    # ============================================================
    # LOAD DATA
    # ============================================================

    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned Titanic dataset:\n{CLEANED_PATH}"
        )

    df = pd.read_csv(CLEANED_PATH)

    print("=" * 80)
    print("TITANIC DATASET - STAGE 2.13: FARE REGRESSION")
    print("=" * 80)

    print("\nCleaned dataset shape:")
    print(df.shape)

    # ============================================================
    # RECREATE STAGE 2.7 SPLIT
    # ============================================================

    print("\nRecreating Stage 2.7 train/test split...")

    # Classification target used for stratification
    y_classification = df["survived"]

    # Remove only the classification target
    X_all = df.drop(
        columns=["survived"]
    )

    # Exact same split settings used in Stage 2.7
    X_train_full, X_test_full = train_test_split(
        X_all,
        test_size=0.20,
        random_state=42,
        stratify=y_classification,
    )

    print(
        f"Training rows: {len(X_train_full)}"
    )

    print(
        f"Test rows: {len(X_test_full)}"
    )

    # ============================================================
    # DEFINE REGRESSION TARGET
    # ============================================================

    y_train = X_train_full["fare"].copy()
    y_test = X_test_full["fare"].copy()

    # ============================================================
    # DEFINE REGRESSION FEATURES
    # ============================================================

    excluded_columns = {
        "fare",
        "survived",
        "alive",
    }

    feature_columns = [
        column
        for column in X_train_full.columns
        if column not in excluded_columns
    ]

    print("\nExcluded columns:")
    print(sorted(excluded_columns))

    print("\nRegression predictor columns:")
    print(feature_columns)

    # ============================================================
    # CREATE X TRAIN / X TEST
    # ============================================================

    X_train = X_train_full[
        feature_columns
    ].copy()

    X_test = X_test_full[
        feature_columns
    ].copy()

    print("\nTraining feature shape:")
    print(X_train.shape)

    print("\nTesting feature shape:")
    print(X_test.shape)

    # ============================================================
    # IDENTIFY FEATURE TYPES
    # ============================================================

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
        ]
    ).columns.tolist()

    print("\nNumeric predictors:")
    print(numeric_features)

    print("\nCategorical predictors:")
    print(categorical_features)

    # ============================================================
    # NUMERIC PREPROCESSING
    # ============================================================

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # ============================================================
    # CATEGORICAL PREPROCESSING
    # ============================================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # ============================================================
    # COLUMN TRANSFORMER
    # ============================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # ============================================================
    # COMPLETE REGRESSION PIPELINE
    # ============================================================

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    # ============================================================
    # TRAIN
    # ============================================================

    print("\n" + "=" * 80)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 80)

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Linear Regression trained successfully."
    )

    # ============================================================
    # PREDICT
    # ============================================================

    predictions = model.predict(
        X_test
    )

    # ============================================================
    # METRICS
    # ============================================================

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    # ============================================================
    # EFFECTIVE NUMBER OF PREDICTORS
    # ============================================================

    fitted_preprocessor = (
        model.named_steps["preprocessor"]
    )

    transformed_feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    transformed_feature_count = len(
        transformed_feature_names
    )

    n = len(y_test)

    p = transformed_feature_count

    adjusted = adjusted_r2(
        r2,
        n,
        p,
    )

    # ============================================================
    # RESIDUALS
    # ============================================================

    residuals = (
        y_test.to_numpy()
        - predictions
    )

    # ============================================================
    # PRINT METRICS
    # ============================================================

    print("\n" + "=" * 80)
    print("REGRESSION METRICS")
    print("=" * 80)

    print(
        f"MAE        : {mae:.4f}"
    )

    print(
        f"RMSE       : {rmse:.4f}"
    )

    print(
        f"R²         : {r2:.4f}"
    )

    print(
        f"Adjusted R²: {adjusted:.4f}"
    )

    print(
        f"Test rows  : {n}"
    )

    print(
        f"Predictors after encoding: "
        f"{transformed_feature_count}"
    )

    # ============================================================
    # HETEROSCEDASTICITY CHECK
    # ============================================================

    prediction_median = np.median(
        predictions
    )

    low_mask = (
        predictions
        <= prediction_median
    )

    high_mask = (
        predictions
        > prediction_median
    )

    low_residuals = residuals[
        low_mask
    ]

    high_residuals = residuals[
        high_mask
    ]

    low_variance = np.var(
        low_residuals,
        ddof=1,
    )

    high_variance = np.var(
        high_residuals,
        ddof=1,
    )

    if low_variance > 0:
        variance_ratio = (
            high_variance
            / low_variance
        )
    else:
        variance_ratio = np.inf

    if (
        variance_ratio > 2
        or variance_ratio < 0.5
    ):
        heteroscedasticity_conclusion = (
            "The residual spread changes noticeably "
            "between lower and higher predicted fares, "
            "which is evidence consistent with "
            "heteroscedasticity."
        )
    else:
        heteroscedasticity_conclusion = (
            "The residual spread is reasonably similar "
            "between lower and higher predicted fares. "
            "This residual-spread check does not show "
            "strong evidence of heteroscedasticity."
        )

    print("\n" + "=" * 80)
    print("RESIDUAL ANALYSIS")
    print("=" * 80)

    print(
        f"Residual mean: "
        f"{residuals.mean():.4f}"
    )

    print(
        f"Residual standard deviation: "
        f"{residuals.std():.4f}"
    )

    print(
        f"Lower predicted-fare residual variance: "
        f"{low_variance:.4f}"
    )

    print(
        f"Upper predicted-fare residual variance: "
        f"{high_variance:.4f}"
    )

    print(
        f"Variance ratio: "
        f"{variance_ratio:.4f}"
    )

    print(
        "\nHeteroscedasticity conclusion:"
    )

    print(
        heteroscedasticity_conclusion
    )

    # ============================================================
    # SAVE METRICS
    # ============================================================

    metrics_df = pd.DataFrame(
        [
            {
                "model": (
                    "Multivariate Linear Regression"
                ),
                "target": "fare",
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Adjusted_R2": adjusted,
                "test_rows": n,
                "transformed_predictor_count":
                    transformed_feature_count,
            }
        ]
    )

    metrics_path = (
        OUTPUT_DIR
        / "fare_regression_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # ============================================================
    # SAVE PREDICTIONS
    # ============================================================

    predictions_df = pd.DataFrame(
        {
            "actual_fare": y_test.to_numpy(),
            "predicted_fare": predictions,
            "residual": residuals,
        }
    )

    predictions_path = (
        OUTPUT_DIR
        / "fare_regression_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # ============================================================
    # RESIDUAL PLOT
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        predictions,
        residuals,
        alpha=0.65,
    )

    ax.axhline(
        y=0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Predicted Fare"
    )

    ax.set_ylabel(
        "Residual (Actual - Predicted)"
    )

    ax.set_title(
        "Fare Regression — "
        "Residuals vs Predicted Fare"
    )

    fig.tight_layout()

    residual_plot = (
        PLOTS_DIR
        / "fare_residual_plot.png"
    )

    fig.savefig(
        residual_plot,
        dpi=150,
    )

    plt.close(fig)

    # ============================================================
    # ACTUAL VS PREDICTED PLOT
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        y_test,
        predictions,
        alpha=0.65,
    )

    lower = min(
        y_test.min(),
        predictions.min(),
    )

    upper = max(
        y_test.max(),
        predictions.max(),
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
    )

    ax.set_xlabel(
        "Actual Fare"
    )

    ax.set_ylabel(
        "Predicted Fare"
    )

    ax.set_title(
        "Fare Regression — "
        "Actual vs Predicted"
    )

    fig.tight_layout()

    actual_predicted_plot = (
        PLOTS_DIR
        / "fare_actual_vs_predicted.png"
    )

    fig.savefig(
        actual_predicted_plot,
        dpi=150,
    )

    plt.close(fig)

    # ============================================================
    # WRITTEN INTERPRETATION
    # ============================================================

    interpretation = f"""
# Stage 2.13 — Fare Regression Interpretation

## Model

A multivariate Linear Regression model was used to predict `fare`
from available passenger features.

The target variable was:

- `fare`

The following variables were excluded from the predictors:

- `fare`, because it is the target
- `survived`, because it is the classification target
- `alive`, because it directly represents the same survival outcome

The regression uses the same train/test partition as Stage 2.7.
All preprocessing is fitted only on the training data using a
scikit-learn Pipeline and ColumnTransformer.

## Preprocessing

Numeric predictors use:

- median imputation
- StandardScaler

Categorical predictors use:

- most-frequent imputation
- OneHotEncoder with `handle_unknown="ignore"`

## Results

- MAE: **{mae:.4f}**
- RMSE: **{rmse:.4f}**
- R²: **{r2:.4f}**
- Adjusted R²: **{adjusted:.4f}**

The test set contains **{n}** observations and the preprocessing
produced **{transformed_feature_count}** predictor columns after
categorical encoding.

MAE represents the average absolute prediction error in fare units.
RMSE gives greater influence to larger prediction errors.
R² measures the proportion of test-set fare variance explained by
the model, while Adjusted R² accounts for the number of predictors.

## Residual analysis

The residual mean is **{residuals.mean():.4f}** and the residual
standard deviation is **{residuals.std():.4f}**.

The residual variance for the lower predicted-fare group is
**{low_variance:.4f}**, while the variance for the higher
predicted-fare group is **{high_variance:.4f}**.

The variance ratio is **{variance_ratio:.4f}**.

### Heteroscedasticity conclusion

{heteroscedasticity_conclusion}

The residual plot should also be inspected visually. A widening
or narrowing residual spread as predicted fare increases would
support the presence of heteroscedasticity, while a reasonably
constant vertical spread would be more consistent with constant
variance.

This stage uses the residual plot and variance comparison as
diagnostics; it does not claim that a formal heteroscedasticity
hypothesis test was performed.

## Data limitation

The earlier EDA showed that `fare` is strongly right-skewed and
contains outliers. Therefore, large-fare observations can have a
substantial influence on RMSE and the residual pattern.
"""

    interpretation_path = (
        OUTPUT_DIR
        / "interpretation.md"
    )

    interpretation_path.write_text(
        interpretation,
        encoding="utf-8",
    )

    # ============================================================
    # SAVE COMPLETE PIPELINE
    # ============================================================

    pipeline_path = (
        OUTPUT_DIR
        / "fare_regression_pipeline.joblib"
    )

    joblib.dump(
        model,
        pipeline_path,
    )

    # ============================================================
    # RELOAD PIPELINE TEST
    # ============================================================

    loaded_model = joblib.load(
        pipeline_path
    )

    sample_prediction = (
        loaded_model.predict(
            X_test.iloc[:1]
        )[0]
    )

    print("\n" + "=" * 80)
    print("PIPELINE RELOAD TEST")
    print("=" * 80)

    print(
        f"Sample raw input prediction: "
        f"{sample_prediction:.4f}"
    )

    # ============================================================
    # SAVED OUTPUTS
    # ============================================================

    print("\n" + "=" * 80)
    print("SAVED OUTPUTS")
    print("=" * 80)

    print(metrics_path)
    print(predictions_path)
    print(interpretation_path)
    print(residual_plot)
    print(actual_predicted_plot)
    print(pipeline_path)

    print("\nStage 2.13 complete.")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()