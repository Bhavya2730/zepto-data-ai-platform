Analytics — Stage 2.12

Random Forest GridSearchCV

This stage tunes Random Forest using 5-fold GridSearchCV with F1 scoring.

Search dimensions:

n_estimators

max_depth

max_features

The estimator is created with oob_score=True as required.

The Stage 2.8 preprocessor was fitted on X_train only. GridSearchCV receives
only transformed training data. The held-out test set is evaluated only after
the best parameters are selected.

Outputs

analytics/results/stage_2_12/
├── gridsearch_summary.csv
├── gridsearch_cv_results.csv
├── tuned_random_forest_test_metrics.csv
└── best_random_forest.joblib

analytics/plots/stage_2_12/
└── top_gridsearch_configurations.png

Run

From the repository root:

python analytics/12_random_forest_gridsearch.py

Final deployment selection is deferred until the required regression and
overall model-comparison stages are complete.