"""
01_tracking.py
--------------------------
Objective 1 - Experiment Tracking

What this does:
  - Trains three different classifiers on the Telco Churn dataset.
  - Logs every run to MLflow: parameters, metrics, and the trained model artifact.
  - Each run is grouped under one MLflow experiment called "Churn_Experiment_Tracking".
"""

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------- MLFLOW SETUP ----------
mlflow.set_experiment("Churn_Experiment_Tracking")

# ---------- LOAD DATA ----------
from preprocess import X_train, X_test, y_train, y_test

# ---------- THREE MODELS TO COMPARE ----------
models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=200, random_state=42), { "solver": "lbfgs", "max_iter": 200 },
    ),
    "DecisionTree": (
        DecisionTreeClassifier(max_depth=5, random_state=42),
        { "max_depth": 5 },
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=100, random_state=42),
        { "n_estimators": 100 },
    )
}

# ---------- TRAIN EACH MODEL IN ITS OWN MLFLOW RUN ----------
for model_name, (model, params) in models.items():
    with mlflow.start_run(run_name=model_name):
        # Log parameters (hyperparameters used for this run)
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Log metrics (accuracy, F1, AUC-ROC)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Log the trained model as an artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[{model_name}] accuracy={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")

print("Experiment tracking complete.")