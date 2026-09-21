Analytics — Stage 2.6

EDA-stage standardization

Stage 2.6 reads:

analytics/titanic_cleaned_stage_2_2.csv

It standardizes the age and fare columns on the full cleaned DataFrame, as required for the exploratory-analysis stage.

The transformation is:

z = (x - mean) / standard deviation

StandardScaler is used to perform the transformation.

Important modeling distinction

This standardization is for EDA only.

It must not be reused as the modeling preprocessing fitted on the full dataset. In the modeling stage, preprocessing will be fitted on the training split only and applied to the test split through a pipeline.

Verification

The script reports the mean and standard deviation before and after standardization.

The transformed variables should have:

mean ≈ 0
standard deviation ≈ 1

StandardScaler uses population standard deviation (ddof=0), while pandas .std() defaults to sample standard deviation (ddof=1). Therefore the printed pandas standard deviation may be slightly above 1; this is expected.

Run

From the repository root:

python analytics/06_eda_standardization.py

The resulting dataset is saved as:

analytics/titanic_eda_standardized.csv