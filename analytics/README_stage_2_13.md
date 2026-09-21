Analytics — Stage 2.13

Fare regression

Stage 2.13 predicts Titanic fare from the other available passenger
features.

The regression target is:

fare

The predictors exclude:

fare
survived

survived is excluded because it is the classification target rather than a
passenger predictor for this regression task.

Preprocessing

A separate regression preprocessing pipeline is fitted on the training data
only:

Numeric features: median imputation + StandardScaler

Categorical features: most-frequent imputation + OneHotEncoder

A Linear Regression estimator is then fitted on the transformed training
data.

Metrics

The stage reports:

MAE

RMSE

R²

Adjusted R²

Adjusted R² uses the number of transformed predictor columns.

Residual analysis

A residual-vs-predicted plot is created to assess whether residual variance
appears approximately constant or whether there is a pattern consistent with
heteroscedasticity.

An actual-vs-predicted plot is also saved.

The interpretation deliberately distinguishes visual diagnosis from a formal
heteroscedasticity statistical test.

Outputs

analytics/results/stage_2_13/
├── fare_regression_metrics.csv
├── fare_regression_predictions.csv
└── interpretation.md

Plots:

analytics/plots/stage_2_13/
├── fare_residual_plot.png
└── fare_actual_vs_predicted.png

Run

From the repository root:

python analytics/13_fare_regression.py