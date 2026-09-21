Analytics — Stage 2.7

Stratified train/test split

Stage 2.7 reads:

analytics/titanic_cleaned_stage_2_2.csv

The target is:

survived

The data is split into:

80% training data

20% test data

random_state=42

stratify=y

Why stratification?

survived is a binary classification target. Stratification preserves approximately the same survivor/non-survivor proportions in both the training and test sets as in the cleaned dataset.

This makes the evaluation split more representative of the original target distribution.

Leakage prevention

Stage 2.7 performs only the split.

It does not:

impute values

encode categorical variables

scale numeric variables

apply SMOTE

fit a classifier

Those operations occur after the split. Modeling preprocessing will be fitted on the training data only.

Saved files

The split datasets are saved under:

analytics/splits/
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
└── class_distribution_comparison.csv

Run

From the repository root:

python analytics/07_train_test_split.py
