# Telco customer churn

CM2604 Machine Learning coursework, Part 01. The project will compare KNN,
logistic regression and random forest for predicting customer churn.

Release 1 provides dataset loading, checksum and schema validation, and a working
inspection command. Models, exploration notebooks and experimental results will
be added in later releases.

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
