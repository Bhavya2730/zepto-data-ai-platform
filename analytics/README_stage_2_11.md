Analytics — Stage 2.11

Class imbalance experiment

This stage evaluates one classifier, Logistic Regression, under three
different approaches to the Titanic target-class imbalance:

Baseline Logistic Regression

Logistic Regression with class_weight="balanced"

Logistic Regression with SMOTE applied only to the training data

All three approaches use the same Stage 2.7 stratified train/test split and
the same Stage 2.8 preprocessing.

Leakage prevention

The preprocessing object was fitted on X_train in Stage 2.8.

For this experiment:

X_train is transformed with that fitted preprocessor.

X_test is transformed with that same preprocessor.

SMOTE is applied only to transformed X_train.

The test set is never oversampled.

The test set is used only for final evaluation.

Required comparison

The main comparison table reports:

Accuracy

Precision

Recall

F1

The experiment also saves confusion-matrix values and a comparison chart for
precision, recall, and F1.

Outputs

analytics/results/stage_2_11/
├── imbalance_comparison.csv
├── confusion_matrices.csv
└── interpretation.md

Plot:

analytics/plots/stage_2_11/
└── imbalance_metric_comparison.png

Run

From the repository root:

python analytics/11_class_imbalance.py

Final classifier deployment selection is intentionally deferred until the
required GridSearchCV and regression/model-comparison stages are complete.