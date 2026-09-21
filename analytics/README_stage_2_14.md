Analytics — Stage 2.14

Final model comparison and deployment recommendation

This stage consolidates the required modeling results from Stages 2.10–2.13.

Classification

The classifier comparison includes:

Logistic Regression

Decision Tree

Random Forest

Tuned Random Forest from GridSearchCV

Metrics:

Accuracy

Precision

Recall

F1

ROC-AUC

The Stage 2.11 imbalance experiment is preserved as a separate comparison
because it evaluates training strategies rather than distinct final model
families.

Regression

Regression metrics are presented separately because fare prediction is a
different task from survival classification.

Reported regression metrics:

MAE

RMSE

R²

Adjusted R²

The regression residual analysis is also summarized.

Final recommendation

The stage writes a 3–5 sentence classifier deployment recommendation based
on the observed metric values, as required by the assignment.

The recommendation does not combine classification and regression metrics
into a single score.

Outputs

analytics/results/stage_2_14/
├── classifier_model_comparison.csv
├── imbalance_strategy_comparison.csv
├── random_forest_gridsearch_summary.csv
├── regression_model_comparison.csv
└── final_model_comparison.md

Run

From the repository root:

python analytics/14_final_model_comparison.py