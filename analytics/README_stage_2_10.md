Analytics — Stage 2.10

Classifier evaluation

Stage 2.10 evaluates the three classifiers trained in Stage 2.9 on the same
held-out test set:

Logistic Regression

Decision Tree

Random Forest

The Stage 2.8 preprocessor is loaded and used to transform the test data. It
is not refitted.

Required metrics

For every classifier:

Confusion matrix

Accuracy

Precision

Recall

F1

ROC-AUC

A single comparison table contains the metrics for all three models.

A combined ROC-curve chart is also created.

Outputs

analytics/results/stage_2_10/
├── classifier_comparison.csv
├── logistic_regression_confusion_matrix.csv
├── decision_tree_confusion_matrix.csv
└── random_forest_confusion_matrix.csv

Plots:

analytics/plots/stage_2_10/
├── logistic_regression_confusion_matrix.png
├── decision_tree_confusion_matrix.png
├── random_forest_confusion_matrix.png
└── roc_curves.png

Run

From the repository root:

python analytics/10_evaluate_classifiers.py

The model comparison is descriptive. Final model-selection reasoning will be
written after the required imbalance experiment and Random Forest
GridSearchCV stages are complete.