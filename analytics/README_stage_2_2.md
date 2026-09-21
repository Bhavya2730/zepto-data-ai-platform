Analytics — Stage 2.2

Dataset profiling

Stage 2.2 reads the committed titanic.csv generated in Stage 2.1.

It reports:

df.info()

df.describe()

df.shape

Missing-value counts and percentages for columns containing missing values.

Missing-value handling

The assignment specifies these thresholds:

Missingness

Handling

<5%

Drop affected rows

5–30%

Impute

>30%

Drop the column or encode missing values as a category

For the Titanic data:

age is in the 5–30% range, so numeric median imputation is used.

embarked is below 5%, so rows missing embarked are dropped.

deck is above 30%, so the column is retained and missing values are encoded as "Unknown". The rationale is that missingness may itself contain information, while dropping the entire feature would discard potentially useful information.

The cleaned output is saved as:

analytics/titanic_cleaned_stage_2_2.csv

Run

From the repository root:

python analytics/02_profile_missing.py

This stage reads the CSV from Stage 2.1. It does not call sns.load_dataset().