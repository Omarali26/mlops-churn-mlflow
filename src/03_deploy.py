"""
03_deploy.py
-----------------------
Objective 3 - Model Deployment

What this does:
  - Loads the best registered model from the MLflow Model Registry.
  - Runs batch predictions on the test set (simulating a batch prediction job).
  - Shows how to do a single real-time prediction (simulating one API call).
  - Saves prediction results to a CSV artifact logged back to MLflow.

In a real deployment you would run:
    mlflow models serve -m "models:/ChurnPredictor/Production" --port 5001
and send POST requests to http://localhost:5001/invocations.
Here we simulate that in Python for simplicity (no extra process needed).
"""

import mlflow
import pandas as pd
import numpy as np

# ---------- MLFLOW SETUP ----------
mlflow.set_experiment("Churn_Deployment")

# ---------- LOAD DATA ----------
from preprocess import X_train, X_test, y_train, y_test

# ---------- LOAD MODEL FROM REGISTRY ----------
try:
    model = mlflow.sklearn.load_model("models:/ChurnPredictor/1")
    print("Loaded ChurnPredictor version 1 from MLflow model Registry")
except Exception as e:
    # Fallback: if registry not populated yet, load the latest run model
    print(f"Registry load failed ({e}), falling back to lates run...")
    runs = mlflow.search_runs(
        experiment_names=["Churn_Hyperparameter_Tuning"],
        order_by=["metrics.f1_score DESC"],
    )

    if runs.empty:
        # Last resort: train a quick model right here
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("No prior runs found; Trained a fresh RandomForest.")
    else:
        run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/best_model"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model from run {run_id}")


# ---------- BATCH PREDICTION ----------
batch_preds = model.predict(X_test)
batch_prebs = model.predict_proba(X_test)[:, 1]

results_df = pd.DataFrame({
    "true_label": y_test,
    "predicted": batch_preds,
    "churn_prob": np.round(batch_prebs, 4),
    "correct": (batch_preds == y_test),
})

results_path = "data\\batch_predictions.csv"
results_df.to_csv(results_path, index=False)
print(f"Batch predictions (first 5 rows): {results_df.head()}")

# ---------- SINGLE REAL-TIME PREDICTION (SIMULATED) ----------
single_input = X_test[0:1] # shape (1, n_features)
single_pred = model.predict(single_input)[0]
single_prob = model.predict_proba(single_input)[0][1]
print("Real-time prediction for one customer:")
print(f"Churn predicted: {"Yes" if single_pred == 1 else "No"}")
print(f"Churn probability: {single_prob:.2%}")

# ---------- LOG DEPLOYMENT RUN IN MLflow ----------
from sklearn.metrics import accuracy_score, f1_score
with mlflow.start_run(run_name="Deployment_BatchRun"):
    mlflow.log_metric("batch_accuracy", accuracy_score(y_test, batch_preds))
    mlflow.log_metric("batch_f1", f1_score(y_test, batch_preds))
    mlflow.log_metric("n_predictions", len(batch_preds))
    mlflow.log_artifact(results_path, artifact_path="predictions")
    
    print("Deployment run logged to MLflow.")

print("Deployment objective complete.")