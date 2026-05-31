"""
04_monitor.py
-----------------------------
Objective 4 - Performance Monitoring

What this does:
  - Simulates 5 "time periods" of incoming data (like weekly batches).
  - Each period has slightly different data distributions to simulate drift.
  - Metrics (accuracy, F1, AUC) are logged to MLflow for each period.
  - A simple drift check flags when performance drops significantly.

In a real system you would run this script on a schedule (e.g., daily cron)
against fresh production data. MLflow keeps the history so you can plot
how metrics change over time in the UI.
"""

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------- MLFLOW SETUP ----------
mlflow.set_experiment("Churn_Performance_Monitoring")

# ---------- LOAD DATA ----------
from preprocess import X_train, X_test, y_train, y_test

# ---------- TRAIN A BASELINE MODEL ON ORIGINAL DATA ----------
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))
print(f"Baseline accuracy: {baseline_acc:.3f}")

def simulate_drift(X, period: int):
    """
    Add small Gaussian noise to features to simulate data drift.
    Noise grows with period number so drift gradually worsens.
    """
    rng = np.random.RandomState(seed=period)
    noise = rng.normal(loc=0, scale=0.05 * period, size=X.shape)
    return np.clip(X + noise, 0, None)

# ---------- MONITOR OVER 5 SIMULATED TIME PERIODS ----------
print("Monitoring performance over time:")
for period in range(1, 6):
    # Simulate drifted data for this period
    X_drifted = simulate_drift(X_test, period)

    y_pred = baseline_model.predict(X_drifted)
    y_prob = baseline_model.predict_proba(X_drifted)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Check for drift: has accuracy dropped beyond threshold?
    drift_detected = (baseline_acc - acc) > 0.05
    drift_flag = 1 if drift_detected else 0

    # Log each period as its own MLflow run
    with mlflow.start_run(run_name=f"Period_{period}"):
        mlflow.log_param("period", period)
        mlflow.log_param("drift_threshold", 0.05)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("accuracy_drop", round(baseline_acc - acc, 4))
        mlflow.log_metric("drift_detected", drift_flag)

    status = "DRIFT DETECTED" if drift_detected else "OK"
    print(f"period: {period}, acc={acc:0.3f}, f1={f1:.3f}, drop={(baseline_acc - acc):.3f}, status={status}")

print("Monitoring complete.")