Analytics — Stage 2.4

Bivariate analysis

Stage 2.4 reads:

analytics/titanic_cleaned_stage_2_2.csv

It does not call sns.load_dataset().

Survival-rate analysis

The script calculates:

Survival rate by sex

Survival rate by pclass

Survival rate by sex + pclass

Boolean masks are explicitly used with & for combined conditions and | for an OR condition.

Required correlation matrix

The correlation matrix contains exactly these six columns:

survived
pclass
age
sibsp
parch
fare

The assignment specifically excludes:

adult_male
alone

The script:

Prints the six-column correlation matrix.

Saves a Seaborn heatmap.

Examines all off-diagonal feature pairs.

Sorts them by absolute correlation.

Reports the two strongest absolute correlations.

Run

From the repository root:

python analytics/04_bivariate_analysis.py

Outputs are saved under:

analytics/plots/stage_2_4/
analytics/results/stage_2_4/

The numerical results will be used for the written interpretation in the final analytics documentation.