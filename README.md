# Zepto Data & AI Platform

A complete data engineering, analytics, machine learning, and GenAI support assistant project built as a three-module capstone.

## Project Overview

The project contains three integrated modules:

1. **Module 1 — Data Pipeline**
2. **Module 2 — Analytics**
3. **Module 3 — Support Assistant**

The project demonstrates data collection and storage, exploratory data analysis, machine learning, model evaluation, and a local Retrieval-Augmented Generation (RAG) support assistant.

---

zepto-data-ai-platform/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_pipeline/
│   ├── scrape_pipeline.py
│   ├── queries.py
│   └── README.md
│
├── analytics/
│   ├── 01_load_titanic.py
│   ├── 02_profile_missing.py
│   ├── 03_univariate_analysis.py
│   ├── 04_bivariate_analysis.py
│   ├── 05_multivariate_story.py
│   ├── 06_standardization.py
│   ├── 07_train_test_split.py
│   ├── 08_preprocessing.py
│   ├── 09_train_classifiers.py
│   ├── 10_evaluate_classifiers.py
│   ├── 11_class_imbalance.py
│   ├── 12_random_forest_gridsearch.py
│   ├── 13_fare_regression.py
│   ├── 14_final_model_comparison.py
│   ├── 15_save_final_pipeline.py
│   ├── README.md
│   ├── titanic.csv
│   ├── titanic_cleaned_stage_2_2.csv
│   ├── titanic_eda_standardized.csv
│   │
│   ├── models/              ← generated/ignored
│   ├── plots/               ← generated/ignored
│   ├── preprocessing/       ← generated/ignored
│   ├── results/             ← generated/ignored
│   └── splits/              ← generated/ignored
│
└── support_assistant/
    ├── 01_verify_corpus.py
    ├── 02_chunk_documents.py
    ├── 03_embed_and_index.py
    ├── 04_retrieve.py
    ├── 05_langgraph_router.py
    ├── 06_structured_graph.py
    ├── 07_test_api.py
    ├── 08_validate.py
    ├── main.py
    ├── retrieval.py
    ├── structured_graph.py
    ├── Dockerfile
    ├── README.md
    ├── chunks.jsonl
    │
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    │
    └── chroma_db/            ← generated/ignored

Technologies Used
Programming
Python
SQL
SQLite
Data Engineering
Requests
BeautifulSoup
SQLite
Pandas
Analytics & Machine Learning
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Imbalanced-learn
Joblib
GenAI / RAG
Sentence Transformers
all-MiniLM-L6-v2
ChromaDB
LangGraph
Pydantic
FastAPI
Uvicorn
Module 1 — Data Pipeline

The first module implements a small data pipeline that collects book information from a public website and stores the cleaned data in SQLite.

Pipeline
Web Source
    ↓
Requests
    ↓
BeautifulSoup
    ↓
Data Extraction
    ↓
Cleaning / Validation
    ↓
CSV Output
    ↓
SQLite Database
    ↓
SQL Queries
    ↓
Pandas Equivalent Queries
Main Files
data_pipeline/
├── scrape_pipeline.py
├── queries.py
└── README.md
scrape_pipeline.py

The scraper:

collects book records
extracts relevant fields
cleans the data
converts GBP prices to INR using the project conversion rate
saves the resulting dataset
creates the SQLite database
queries.py

The query module demonstrates:

SQL queries
Pandas equivalents
aggregation
filtering
joins / analytical queries
Running Module 1

From the project root:

python data_pipeline/scrape_pipeline.py

Then:

python data_pipeline/queries.py
Module 2 — Analytics

Module 2 performs exploratory data analysis, preprocessing, classification, imbalance handling, hyperparameter tuning, regression, and final model serialization using the Titanic dataset.

Pipeline
Titanic Dataset
      ↓
Data Profiling
      ↓
Missing Value Handling
      ↓
Outlier Analysis
      ↓
EDA
      ↓
Standardization
      ↓
Stratified Train/Test Split
      ↓
Preprocessing Pipeline
      ↓
Classification Models
      ↓
Evaluation
      ↓
Imbalance Handling
      ↓
Hyperparameter Tuning
      ↓
Regression Analysis
      ↓
Model Comparison
      ↓
Final Pipeline
Main Stages
Data Loading

The Titanic dataset is loaded and saved locally as:

analytics/titanic.csv
Missing Values

The project analyzes missingness and applies appropriate handling strategies including:

median imputation
dropping affected rows where appropriate
explicit Unknown handling for the high-missingness deck feature
Exploratory Analysis

The analysis includes:

distributions
outliers
survival rates
group comparisons
correlations
multivariate visualizations
Preprocessing

The preprocessing pipeline includes:

numerical median imputation
categorical most-frequent imputation
standardization
one-hot encoding
fitting preprocessing only on training data
Classification

Three baseline models are evaluated:

Logistic Regression
Decision Tree
Random Forest

The project also includes:

class-weight balancing
SMOTE
Random Forest GridSearchCV
OOB evaluation
comparison of model metrics
Regression

A multivariate regression model is used to predict fare and evaluated using:

MAE
RMSE
R²
Adjusted R²
residual analysis
Running Module 2

The scripts are numbered according to the analysis sequence.

For example:

python analytics/01_load_titanic.py
python analytics/02_profile_missing.py

Continue through:

python analytics/15_save_final_pipeline.py

The individual scripts document their respective outputs and analysis stages.

Module 3 — Support Assistant

Module 3 implements a local policy-grounded GenAI support assistant.

The assistant uses:

8 policy documents
document chunking
local Sentence Transformer embeddings
ChromaDB
LangGraph
Pydantic structured output
FastAPI
deterministic offline mock behavior
RAG Architecture
Policy Documents
      ↓
Chunking
      ↓
Local Embeddings
      ↓
ChromaDB
      ↓
User Query
      ↓
LangGraph Intent Routing
      ↓
Policy Retrieval / Direct Answer
      ↓
Pydantic Structured Response
      ↓
FastAPI JSON
Knowledge Base

The knowledge base contains 8 policy documents covering:

Delivery Policy
Returns & Refunds
Membership Tiers
Order Tracking
Order Cancellation Policy
Damaged or Missing Items
Gift Cards
Customer Support Hours

The documents are divided into 22 chunks.

Embeddings

The project uses:

all-MiniLM-L6-v2

Embedding generation is local and does not require an external embedding API.

ChromaDB

ChromaDB is used as the local vector store.

The retriever returns the top 3 most relevant chunks for policy queries using cosine similarity.

LangGraph

The graph contains three main nodes:

classify_intent
       │
       ├── policy_question
       │        ↓
       │  retrieve_and_answer
       │
       └── general_question
                ↓
          direct_answer

Policy routing uses deterministic keyword-based classification in the default mock path.

Structured Response

Responses follow a Pydantic schema:

{
  "answer": "string",
  "sources": ["chunk_id"],
  "confidence": 1.0
}
Mock Mode

The graded baseline runs in deterministic offline mock mode.

The default setting is:

MOCK_LLM=1

No external LLM API key is required for the default path.

Running Module 3

From the project root:

python support_assistant/01_verify_corpus.py

Then:

python support_assistant/02_chunk_documents.py
python support_assistant/03_embed_and_index.py
python support_assistant/04_retrieve.py
python support_assistant/05_langgraph_router.py
python support_assistant/06_structured_graph.py

Start the FastAPI service with:

uvicorn support_assistant.main:app --reload

The API is available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

Test the API from another terminal:

python support_assistant/07_test_api.py

Expected result:

Stage 3.7 API verification passed.
End-to-End Validation

Run:

python -m support_assistant.08_validate

Expected result:

ALL STAGE 3.8 CHECKS PASSED
Docker

A Dockerfile is provided for the Support Assistant.

From the project root, build the image with:

docker build -f support_assistant/Dockerfile -t zepto-support-assistant .

Run the container with:

docker run -p 8000:8000 zepto-support-assistant

The API can then be accessed at:

http://127.0.0.1:8000
API Example
Request
POST /ask
{
  "query": "What is the delivery fee?"
}
Response
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_01_chunk_00"
  ],
  "confidence": 1.0
}
General Query
{
  "query": "What is the capital of France?"
}

Response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
Setup

Clone the repository and enter the project directory:

git clone <repository-url>
cd zepto-data-ai-platform

Install the dependencies:

pip install -r requirements.txt

The project uses Python 3.10+.

Design Decisions
Local and Reproducible

The Support Assistant uses local embeddings and a local ChromaDB instance so that the graded baseline does not depend on an external vector database or API.

Deterministic Mock Mode

The default mock path provides reproducible behavior and does not require an API key.

Modular Design

Each module is separated into its own directory and contains numbered scripts so that individual stages can be executed and verified independently.

Reusable Preprocessing

The analytics module uses a scikit-learn preprocessing pipeline so that the same transformations can be applied consistently during model training and prediction.

Structured Responses

Pydantic is used to validate Support Assistant responses and enforce the required output structure.

Git Workflow

The repository follows the required Git workflow:

Feature Branch
      ↓
Multiple Commits
      ↓
Merge into main
      ↓
Push to GitHub

The final repository is maintained on the main branch.

Final Validation

The three modules were implemented and tested independently.

Module 1 — Data Pipeline        ✅
Module 2 — Analytics            ✅
Module 3 — Support Assistant    ✅
Git Workflow                    ✅
Dockerfile                      ✅
Documentation                   ✅
End-to-End Validation           ✅

The project is designed to provide a complete demonstration of data engineering, analytics and machine learning, and a local RAG-based GenAI application.


  