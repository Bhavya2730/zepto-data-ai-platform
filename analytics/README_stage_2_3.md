Analytics — Stage 2.3

Univariate analysis

Stage 2.3 reads:

analytics/titanic_cleaned_stage_2_2.csv

It does not call sns.load_dataset().

Required analysis

The script produces:

Histogram of age

Box plot of age

Histogram of fare

Box plot of fare

IQR outlier counts for age and fare

Fare mean

Fare median

Fare mode

Fare skewness conclusion based on the ordering of mean, median, and mode

IQR method

For each numeric variable:

IQR = Q3 - Q1

Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR

Any value outside those bounds is counted as an outlier.

Run

From the repository root:

python analytics/03_univariate_analysis.py

Plots are saved under:

analytics/plots/stage_2_3/

The generated plots are analysis artifacts for the module and should be referenced from the final analytics README/notebook where appropriate.