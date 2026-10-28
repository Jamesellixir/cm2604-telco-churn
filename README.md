# Telco customer churn

CM2604 Machine Learning coursework, Part 01. The project will compare KNN,
logistic regression and random forest for predicting customer churn.

Release 5 adds the approved grid searches, compares default/tuned validation
results and freezes a configuration before final test evaluation. Both notebooks
contain executed results. The held-out test partition remains unevaluated.

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
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" -m src.train
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" -m src.train --defaults
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" -m src.train --tune
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" scripts/run_notebook.py 02_model_comparison.ipynb
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
The held-out features are reserved for final evaluation. The exploration notebook
describes the training data before imputation and scaling.

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

## Preprocessing and baseline

`build_preprocessor(scaled=True)` constructs an unfitted transformer for KNN and
logistic regression. `scaled=False` supplies unscaled numeric values for random
forest and the dummy baseline. The three numeric fields are converted to numbers;
blanks become missing values. Missing numbers use training-fold medians, followed
by numeric standardisation in the scaled branch. The other 16 predictors,
including SeniorCitizen, use most-frequent imputation and one-hot encoding.
Unseen categories produce zeros in that field's indicator columns. Service labels
such as "No internet service" remain separate categories.

The source loader validates the original CSV strictly. The preprocessing functions
also handle missing predictor values in new inputs. A numeric column that is
entirely missing in a training fold is retained and filled with zero as a fallback.
No source rows are dropped, resampled or used to fit preprocessing before the split.

The baseline pipeline combines preprocessing with
`DummyClassifier(strategy="most_frequent")`. It learns the majority class from
each training fold and always predicts that class. This is a reference for later
models; it does not use customer attributes to choose its predictions
(scikit-learn developers, n.d.a).

`python -m src.train` uses only the 5,634 training rows with five stratified folds,
shuffling and seed 42. Each fold clones and fits the complete pipeline, including
its imputation and encoding. The command exports
[fold results](outputs/tables/baseline_cv_folds.csv),
[means and standard deviations](outputs/tables/baseline_cv_summary.csv) and
[configuration](outputs/tables/baseline_configuration.json).
Times are measured in seconds and vary between runs. Standard deviations use
`ddof=0`; all five folds have equal weight. Churn (`Yes = 1`) is the positive class.
Undefined precision is reported as zero using `zero_division=0`.

| Cross-validation metric | Mean | Standard deviation |
|---|---:|---:|
| Churn F1 | 0.000000 | 0.000000 |
| Churn precision | 0.000000 | 0.000000 |
| Churn recall | 0.000000 | 0.000000 |
| Accuracy | 0.734647 | 0.000094 |
| Balanced accuracy | 0.500000 | 0.000000 |
| ROC-AUC | 0.500000 | 0.000000 |
| Average precision | 0.265353 | 0.000094 |

Every fold predicts No for every validation customer. Its approximately 73.46%
accuracy therefore comes with zero churn recall. This gives the later models a
reference that accuracy alone would hide. The `most_frequent` strategy assigns
churn probability zero, so its ROC-AUC is 0.5 and its average precision equals
the validation churn fraction. These are training cross-validation results;
the held-out test partition has not been evaluated.

Tests check hand-calculated imputation and scaling, missing values, unknown
categories, reproducible folds and aligned labels. Fold audits inspect the fitted
transformers to verify that medians, scaling and category vocabularies come from
that fold's training rows. See scikit-learn's guidance on keeping preprocessing
inside pipelines to prevent leakage (scikit-learn developers, n.d.b).

References: scikit-learn developers (n.d.a)
[*DummyClassifier*](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html);
scikit-learn developers (n.d.b)
[*Common pitfalls and recommended practices*](https://scikit-learn.org/stable/common_pitfalls.html).

## Default classifier comparison

Open [the executed comparison notebook](notebooks/02_model_comparison.ipynb)
for short model explanations, checked fold membership and the default/tuned
comparison. The notebook calls the shared Python modules and reads the saved
search results. `--defaults` reproduces the initial default tables; `--tune`
reproduces the searches and frozen configuration. Run the notebook command last
to independently replay the six frozen settings and refresh the displayed results.

| Classifier | Starting settings | Numeric preprocessing |
|---|---|---|
| KNN | Five neighbours vote equally; Euclidean distance | Median imputation and standardisation |
| Logistic regression | C=1, lbfgs, max_iter=2000, no class weights | Median imputation and standardisation |
| Random forest | 100 trees, unrestricted depth, seed 42 | Median imputation without scaling |

These are the starting settings fixed in the plan, including the larger iteration
limit for logistic regression. Each model learns its preprocessing separately
within the same five folds. All use the same seven metrics and normal binary
decision boundary at 0.5, with ties favouring No. The initial default comparison
performs no tuning or threshold adjustment.

| Model | Mean churn F1 | Mean churn recall | Mean accuracy | Mean ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline | 0.0000 | 0.0000 | 0.7346 | 0.5000 |
| KNN | 0.5457 | 0.5311 | 0.7655 | 0.7822 |
| Logistic regression | 0.5930 | 0.5431 | 0.8026 | 0.8461 |
| Random forest | 0.5385 | 0.4749 | 0.7842 | 0.8178 |

Logistic regression has the highest mean churn F1 among the default
configurations. The tuning stage below determines the frozen choice across
both settings. These scores concern training cross-validation only.

The comparison exports [per-fold scores](outputs/tables/default_cv_folds.csv),
[means and standard deviations](outputs/tables/default_cv_summary.csv),
[model parameters](outputs/tables/default_model_configuration.json) and
[a score chart](outputs/figures/default_model_metrics.png). Existing Release 3
baseline files are preserved. Timings vary between runs; metric scores should
reproduce in the pinned environment. The notebook audits preprocessing statistics,
probability ranges and metric calculations without exporting customer-level rows.

## Fixed-grid tuning and frozen choice

```powershell
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" -m src.train --tune
& "$env:USERPROFILE\venvs\cm2604-telco-churn\Scripts\python.exe" scripts/run_notebook.py 02_model_comparison.ipynb
```

The three approved grids contain 16 KNN, 8 logistic regression and 16 random
forest combinations. GridSearchCV uses the same five training folds, all seven
metrics and `refit="f1"`; preprocessing remains inside each pipeline. This runs
200 validation fits and three training-only refits. No additional searches or
threshold changes are made. Full candidate scores, fold scores, ranks and timing
summaries are retained in each model's `*_search_results.csv` under outputs/tables.

| Model | Best search parameters | Default F1 | Tuned F1 |
|---|---|---:|---:|
| KNN | 31 neighbours, uniform weights, p=1 | 0.5457 | 0.6069 |
| Logistic regression | C=0.1, balanced class weights | 0.5930 | 0.6289 |
| Random forest | 100 trees, depth unrestricted, minimum leaf size 5, balanced class weights | 0.5385 | 0.6354 |

The frozen choice is **tuned random forest**, with mean training CV churn F1
0.6354 and population standard deviation 0.0124. It has the largest mean F1
among the six configurations. Exact model ties favour logistic regression, then
KNN, then random forest; within one model an exact tie prefers the default.
Within a search, the first tied candidate in deterministic grid order wins.

[The frozen configuration](outputs/frozen_configuration.json) stores all six
reconstructible model settings, the selected ID, decision threshold, data checksum,
training-index hash and selection rule. A rerun accepts an identical freeze and
refuses to overwrite a changed one. It stores parameters, not a model binary.
[Tuning metadata](outputs/tables/tuning_configuration.json) records grids and
chosen parameters; [the comparison table](outputs/tables/default_tuned_comparison.csv)
includes every metric and timing for both settings.

The notebook checks all 40 candidates and independently re-evaluates the six
frozen configurations, including fold-only preprocessing and probability checks.
It does not rerun the full grids; use `--tune` to reproduce those searches first.
Scores and choices should reproduce with unchanged data and pinned software;
timings vary. Tuning scores helped choose parameters and may be optimistic.
The held-out test set must evaluate these frozen choices without further tuning.
Final test evaluation, ethics/error analysis and the report remain future work.

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
