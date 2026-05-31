# AIN-3009 - MLOps Term Project
## Telco Customer Churn Prediction with MLflow
---

## Project Overview

This project predicts whether a telecom customer will leave (churn) using a Random Forest classifier trained on the Telco Customer Churn dataset. MLflow is used throughout to manage every stage of the ML lifecycle.

---

## 1. Domain and Dataset

**Domain:** Telecommunications - Customer Churn Prediction
**Dataset:** `data\telco-customer-churn.csv` (7,043 customers, 19 features)

The goal is to predict whether a customer will leave (churn = Yes/No) based on their account information: contract type, monthly charges, tenure, internet service, and so on. Churn prediction is a classic and practical ML problem in telecom - retaining existing customers costs far less than acquiring new ones.

**Why this dataset:**
- Clear binary classification task (Churn: Yes/No)
- Mix of numeric and categoriccal features - realistic preprocessing needed
- Imbalaned classes (~26% churn) - good for evaluating F1 and AUC, not just accuracy

**Preprocessing (`src\preprocess.py`)**
- Drop `customerID` (identifier, not a feature)
- Label-encode all categorical columns (Yes/No -> 1/10, contract type -> integer, etc.)
- Stratified 80/20 train/test split (preserves churn ratio in both sets)

---

## 2. MLflow Setup

```bash
pip install mlflow optuna
```

**Tracking URI:** `mlflow.db`

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Experiment Name") # each objective has it own name

Launch MLflow with port 5000 or change port number you specified:
```bash
mlflow server --port 5000
```

---

## 2. Objectives

### Objective 1 - Experiment Tracking (`src\01_tracking.py`)

Three classifiers are trained and every run is logged under the experiment
`Churn_Experiment_Tracking`:

| Model | Parameters logged | Metrics logged |
| --- | --- | --- |
| Logistic Regression | `max_iter`, `solver` | accuracy, F1, AUC-ROC |
| Decision Tree | `max_depth` | accuracy, F1, AUC-ROC |
| Random Forest | `n_estimators` | accuracy, F1, AUC-ROC |

**How MLflow tracking works:**
- `mlflow.start_run()` opens a run context
- `mlflow.log_params(dict)` records hyperparameters
- `mlflow.log_metric(name, value)` records evaluation scores
- `mlflow.sklearn.log_model(model, ...)` saves the trained model as an artifact

In the MLflow dashboard(UI) `Churn_Experiment_Tracking`, you can select all three runs and compare their metrics side-by-side in a table or parallel coordinates plot.

---

### Objective 2 - Model Training and Hyperparameter Tuning (`src\02_training_tuning.py`)

**Tool used:** Optuna with Tree of Parzen Estimators (TPE) algorithm.

**How it works:**
- Optuna calls the `objective()` function repeatedly (10 trials)
- Each trail trains a model and returns F1 (Optuna maximises)
- TPE uses past results to intelligently pick the next parameter combination
- Each trail is logged as a nested child run inside MLflow

After the search, the best parameters are decoded and the model is re-trained and registered in the MLflow Model Registry as `ChurnPredictor`.

---

### Objective 3 - Model Deployment (`src\03_deploy.py`)

The best model is loaded from the MLflow Model Registry:
```python
model = mlflow.sklearn.load_model("model:/ChurnPredictor/1")
```

**Batch prediction:** The entire test set (200 rows) is scored at once and results are saved to `batch_predictions.csv`, which is logged as an MLflow artifact under the `Churn_Deployment` experiment.

**Real-time (single) prediction:** One customer record is scored individually, returning a churn prediction and probability - this simulates a single API call.

**To serve as an HTTP API** (optional, for demo):
```bash
mlflow models serve -m "models:/ChurnPredictor/Production" --port 5001
# POST to http://localhost:5001/invocations with JSON input
```

---

## Objective 4 - Performance Monitoring (`src\04_monitor.py`)

Five simulated time periods are created by adding increasing Gaussian noise to the test features - this simulates real-world data drift where incoming production data gradually differs from training data.

for each period, accuracy, F1, and AUC are are logged as a separate MLflow run under `Churn_Performance_Monitoring`. Two extra metrics are tracked:

- `accuracy_drop`: how much accuracy has falled vs. baseline
- `drift_detected`: 1 if drop excedds the threshold (0.05), else 0

**To visualise drift in the MLflow UI:**
1. Open `Churn_Performance_Monitoring` experiment
2. Click "Chart" tab -> add "accuracy" metric -> view across runs

In a production system this script would run on a schedule (e.g., daily cron) against real incoming data.

---

## Objective 5 - Model Registry (`src\05_model_registry.py`)

Two model versions are trained and registered:

| Version | Config | Stage |
| --- | --- | --- |
| v1 (Baseline) | 50 trees, depth 5 | Archived |
| v1 (Tuned) | 150 trees, depth 7, min_split 4 | Production |

**Stage transition workflow:**
```
None -> Staging -> Production -> Archived
```

Transitions are performed via the `MlflowClient`:
```python
client.transition_model_version_stage(name="ChurnPredictor", verison=2, stage="Production")
```

In the MLflow UI under **Models -> ChurnPredictor**, each version is visible with its current stage and the run that produced it.

---

## 3. Results Summary

| Model | Accuracy | AUC-ROC |
| --- | --- | --- |
| Logistic Regression | 0.803 | 0.840 |
| Decision Tree | 0.780 | 0.819 |
| Random Forest (baseline) | 0.789 | 0.823 |
| Random Forest (tuned, best) | 0.800 | 0.837 |

The tuned Random Forest achieved the best overall performance and was promoted to Production in the Model Registry.

---

## 4. How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Omarali26/mlops-churn-mlflow.git

# 2. Install dependencies
pip install mlflow optuna

# 3. Run all objectives by (run main.py)
python main.py

# 4. Open the MLflow server
mlflow server --port 5000
```

---

## 5. Folder Structure

```
PRJ-omar_ali-2104642/
|--- data/
|       telco-customer-churn.csv
|--- mlruns/                            # auto-generated
|--- src/
|       preprocess.py
|       01_tracking.py                  # Objective 1
|       02_training_tuning.py           # Objective 2
|       03_deploy.py                    # Objective 3
|       04_monitor.py                   # Objective 4
|       05_model_registry.py            # Objective 5
|--- main.py
|--- mlflow.db                          # auto-generated
|--- README.md
|--- requirements.txt
```
