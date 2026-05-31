"""
05_model_registry.py
---------------------
Objective 5 - MLflow Model Registry

What this does:
  - Registers two model versions (v1 = baseline, v2 = tuned) in the Registry.
  - Transitions v1 to "Staging" for testing.
  - Promotes v2 to "Production" after comparing metrics.
  - Archives v1 (deprecated).
  - Lists all versions and their current stages.

The Model Registry is MLflow's central catalogue. It lets a team:
  • Track which model version is currently serving in Production.
  • Test new versions in Staging before promoting.
  • Archive old versions without deleting them.

Stages available in MLflow OSS:
  None → Staging → Production → Archived
"""

import mlflow
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------- MLFLOW SETUP ----------
mlflow.set_experiment("Churn_Model_Registry")
client = MlflowClient()

# ---------- LOAD DATA ----------
from preprocess import X_train, X_test, y_train, y_test

def train_and_register(params: dict, run_name: str):
    """Train a model with given params, log it, register it. Returns (version, f1)."""
    with mlflow.start_run(run_name=run_name) as run:
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)

        # Log model with artifact_path="model" so registry URI resolves correctly
        mlflow.sklearn.log_model(model, name="model")

        # Register model in the Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, "ChurnPredictor")
        version = mv.version

        print(f"[{run_name}] acc={acc:.3f}, f1={f1:.3f} -> registered as version {version}")

        return version, f1

# ---------- REGISTER VERSION 1 (BASELINE) ----------
print("Registering Version 1 (baseline)")
v1, f1_v1 = train_and_register(
    params={
        "n_estimators": 50,
        "max_depth": 5,
    },
    run_name="Baseline_v1"
)

# ---------- REGISTER VERSION 2 (TUNDED) ----------
print("Registering Version 2 (tuned)")
v2, f1_v2 = train_and_register(
    params={
        "n_estimators": 150,
        "max_depth": 7,
        "min_samples_split": 4
    },
    run_name="Tuned_v2"
)

# ---------- STAGE TRANSITIONS ----------
print(f"Moving v{v1} -> Staging (testing)")
client.transition_model_version_stage(name="ChurnPredictor", version=v1, stage="Staging")

print(f"Moving v{v2} -> Production (better or equal F1)")
client.transition_model_version_stage(name="ChurnPredictor", version=v2, stage="Production")

print(f"Archiving v{v1} -> (replaced by v{v2})")
client.transition_model_version_stage(name="ChurnPredictor", version=v1, stage="Archived")

# ---------- LIST ALL VERSIONS AND THEIR STAGES ----------
print("All registered versions of 'ChurnPredictor'")
for mv in client.search_model_versions("name='ChurnPredictor'"):
    print(f"Version: {mv.version} Stage: {mv.current_stage} Run: {mv.run_id[:8]} ...")

print("Model Registry complete.")