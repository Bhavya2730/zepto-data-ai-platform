Analytics — Stage 2.5

Multivariate data story

Stage 2.5 reads:

analytics/titanic_cleaned_stage_2_2.csv

It creates four distinct charts, each combining multiple variables:

Survival Rate by Passenger Class and Sex

Variables: survived, pclass, sex

Age Distribution by Sex and Survival

Variables: age, sex, survived

Fare vs Age by Survival and Passenger Class

Variables: fare, age, survived, pclass

Survival Rate by Family Size and Passenger Class

Variables: derived family_size, pclass, survived

Each chart has a 2–4 sentence interpretation saved in:

analytics/results/stage_2_5/interpretations.md

The interpretations are deliberately descriptive rather than claiming causation.

Run

From the repository root:

python analytics/05_multivariate_story.py

Charts are saved under:

analytics/plots/stage_2_5/

Supporting summary tables are saved under:

analytics/results/stage_2_5/