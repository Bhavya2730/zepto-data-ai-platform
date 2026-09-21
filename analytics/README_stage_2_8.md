Analytics — Stage 2.8

Training-only preprocessing

Stage 2.8 reads the split files created in Stage 2.7:

analytics/splits/X_train.csv
analytics/splits/X_test.csv

The target is not included because survived was separated in Stage 2.7.

Preprocessing

Numeric features

pclass
age
sibsp
parch
fare

Processing:

Median imputation

StandardScaler

Categorical features

sex
embarked
deck

Processing:

Most-frequent imputation

OneHotEncoder(handle_unknown="ignore")

Leakage prevention

The ColumnTransformer is fit only on X_train:

preprocessor.fit_transform(X_train)

The test data is only transformed:

preprocessor.transform(X_test)

No test data is used to calculate imputation values, scaling parameters, or category mappings.

Feature selection note

The original Titanic dataset contains several derived/redundant variables such as alive, class, who, adult_male, and alone. This stage intentionally uses the core explanatory variables listed above rather than including variables that can duplicate target information or provide redundant representations.

deck is retained because Stage 2.2 explicitly handled its high missingness by encoding missing values as "Unknown".

Saved outputs

analytics/preprocessing/
├── X_train_transformed.csv
├── X_test_transformed.csv
└── preprocessor.joblib

Run

From the repository root:

python analytics/08_preprocessing.py