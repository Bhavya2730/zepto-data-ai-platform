Analytics — Stage 2.9

Three classifier models

Stage 2.9 trains three classifiers using the same Stage 2.7 split and the
same training-fitted preprocessing from Stage 2.8:

Logistic Regression

Decision Tree

Random Forest

The preprocessing object from:

analytics/preprocessing/preprocessor.joblib

is loaded and used for both train and test transformations. It is not
refitted.

Decision Tree visualization

The Decision Tree is rendered with:

transformed feature names

class names:

Did not survive

Survived

The tree image is saved to:

analytics/plots/stage_2_9/decision_tree.png

Saved models

Models are saved under:

analytics/models/
├── logistic_regression.joblib
├── decision_tree.joblib
└── random_forest.joblib

Evaluation

Stage 2.9 intentionally performs only model training and a basic prediction
sanity check. Full evaluation — confusion matrices, accuracy, precision,
recall, F1, ROC/AUC, and the comparison table — is handled in Stage 2.10.

Run

From the repository root:

python analytics/09_train_classifiers.py