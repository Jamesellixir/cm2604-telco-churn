# Telco customer churn

CM2604 Machine Learning coursework, Part 01. The project will compare KNN,
logistic regression and random forest for predicting customer churn.

Release 2 adds the fixed stratified split and training-data exploration to the
dataset validation from Release 1. The executed notebook contains aggregate
statistics, charts and supported observations. Models and experimental results
will be added in later releases.

## Setup

Use Python 3.12. Create the environment outside OneDrive. From the repository root
in PowerShell:

```powershell
py -3.12 -m venv "$env:USERPROFILE\venvs\cm2604-telco-churn"
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" -m pip install -r requirements.txt
```

The requirements pin the tested environment, including dependencies for later
notebooks and modelling. Activation is optional: commands below use the interpreter
directly. No service runs in the background.

## Dataset

Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and extract `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/source/`.
Create those directories if needed. The CSV is excluded from Git because
redistribution rights have not been confirmed.

Expected CSV SHA-256:

```text
88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a
```

The original archive used for this project has SHA-256
`cf7e6dcd8a238ecaa841a7d133142525453992d8d5e3ef6d1e5f0d359e7bf444`.
The command verifies the CSV bytes before parsing. A different download or changed
line endings will fail the checksum; retain the original file rather than editing it.

## Run and test

```powershell
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" -m src.data
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" -m unittest discover -s tests -v
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" scripts/run_exploration.py
```

Use `-m src.data --path "C:\path\to\WA_Fn-UseC_-Telco-Customer-Churn.csv"`
for another location. Run commands from the repository root. Validation failures
print `Data error:` and return exit code 1.

The original CSV contains 7,043 rows and 21 columns: 5,174 `No` and 1,869 `Yes`
churn labels. There are 11 missing `TotalCharges` values and no duplicate customer
IDs. Inspection prints these structural counts without customer-level records.

The loader checks exact, unique columns, unique nonblank identifiers, allowed
categories and target labels, finite non-negative numeric values and whole-month
tenure. Blank `TotalCharges` becomes a missing numeric value. Imputation is deferred
to training pipelines. `separate_inputs_target` removes `customerID`, returns the
19 predictors, and maps `No` to 0 and `Yes` to 1. Tests use synthetic records.

## Training-data exploration

Open [the executed exploration notebook](notebooks/01_data_exploration.ipynb)
on GitHub or in VS Code. To run it interactively, select the Python environment
created above, restart the kernel and run all cells from top to bottom. The
runner command also starts a fresh kernel using its own Python interpreter;
it needs no globally installed notebook kernel. It rewrites the executed
notebook and regenerates the aggregate outputs.

`split_data` returns `X_train, X_test, y_train, y_test`. It uses stratification by
churn, seed 42 and a 20% test fraction. The original CSV order must stay unchanged.
The resulting partitions contain 5,634 training rows and 1,409 held-out rows.
The held-out features are reserved for final evaluation. Imputation and scaling
are deferred to later training pipelines.

Seven aggregate CSV tables are in [outputs/tables](outputs/tables), including
numeric statistics, missing values, categorical counts and churn comparisons.
Four charts are in [outputs/figures](outputs/figures): churn balance, numeric
distributions, churn by contract and numeric features by churn.

In the training partition, 26.54% of customers churned. Eight TotalCharges values
are missing, all at zero tenure. Observed churn rates are 42.75% for month-to-month,
11.08% for one-year and 2.87% for two-year contracts. Median tenure is 10 months
for churned customers and 38 months for non-churned customers. These are
descriptive associations within the training data, not estimates of causation
or model performance.

## Coursework requirements

| Requirement | Planned evidence |
|---|---|
| Corpus preparation (25%) | Validation, training-data exploration and preprocessing |
| Implementation (25%) | KNN, logistic regression, random forest and tuning |
| Evaluation (25%) | Shared validation folds and held-out test comparisons |
| Ethics (10%) | Development choices and post-deployment recommendations |
| Limitations (5%) | Evidence-based limitations and enhancements |
| Viva (10%) | Individual explanation of the completed work |

The final report has a 2,000-word limit and must include the public Git URL and
all source code as text in its appendix. The assignment deadline is 15 December
2026; the time is unspecified. Questions remain about the required report file
format and the meaning of "both model experimental settings". PDF and default
versus tuned models are provisional choices. Dataset redistribution permission
also remains unresolved. The separate closed-book test is outside this project.
