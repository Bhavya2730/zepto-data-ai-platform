Analytics — Stage 2.15

Save and reload the final classification pipeline

Stage 2.15 persists the final classifier selected in Stage 2.14.

The final estimator is the tuned Random Forest:

n_estimators = 200
max_depth = 10
max_features = None
random_state = 42
oob_score = True

Complete pipeline

A single sklearn Pipeline contains:

Raw-input preprocessing

The tuned Random Forest classifier

Numeric columns:

pclass, age, sibsp, parch, fare

are median-imputed and standardized.

Categorical columns:

sex, embarked, deck

are most-frequent-imputed and one-hot encoded with
handle_unknown="ignore".

Persistence test

The complete pipeline is saved with joblib.dump(), reloaded with
joblib.load(), and used to predict directly from raw input.

The raw-input test includes a missing age value to demonstrate that the
saved pipeline performs preprocessing automatically at inference time.

Outputs

analytics/models/
└── final_tuned_random_forest_pipeline.joblib

analytics/results/stage_2_15/
├── reloaded_pipeline_test_predictions.csv
├── raw_input_prediction.csv
└── pipeline_specification.md

Run

From the repository root:

python analytics/15_save_final_pipeline.py

This completes the required Module 2 model-persistence stage.