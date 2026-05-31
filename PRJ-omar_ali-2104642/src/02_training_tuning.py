"""
02_training_tuning.py
----------------------------
Objective 2 - Model Training and Hyperparameter Tuning

What this does:
    - Uses Optuna to search the best hyperparameters for a Random Forest.
    - Every single trial is logged as a child run inside MLflow.
    - After the search, the best parameters are re-trained and logged as the final "best" run and registered in the Model Registry.
    
Optuna works by defining a search space inside an objective function and
running multiple trails. It uses the TPE sampler (same algorithm as Hyperopt)
to intelligently pick the next parameter combination based on past results.    
"""

import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# ---------- MLFLOW SETUP ----------
mlflow.set_experiment("Churn_Hyperparameter_Tuning")

# ---------- LOAD DATA ----------
from preprocess import X_train, X_test, y_train, y_test

# ---------- OPTUNA OBJECTIVE FUNCTION - CALLED ONCE PER TRIAL ----------
def objective(trial):
    """Train RF with suggested params; return F1 for Optuna to maximise."""
    params = {
        "n_estimators":         trial.suggest_categorical("n_estimators", [50, 100, 150, 200]),
        "max_depth":            trial.suggest_categorical("max_depth", [3, 5, 7, 10, None]),
        "min_samples_split":    trial.suggest_int("min_samples_split", 2, 10),
    }

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)

        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 =  f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc_roc", auc)

        print(f"Trial {trial.number}: params={params}, f1={f1:.3f}")

    return f1

# ---------- RUN THE SEARCH ----------
optuna.logging.set_verbosity(optuna.logging.WARNING)

with mlflow.start_run(run_name="OptunaSearch_Parent"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

best_params = study.best_params
print(f"\nBest params: {best_params}")

# ---------- RE-TRAIN WITH THE BEST PARAMS AND LOG AS FINAL MODEL ----------
with mlflow.start_run(run_name="BestModel_Retrained"):
    mlflow.log_params(best_params)

    final_model = RandomForestClassifier(random_state=42, **best_params)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc_roc", auc)

    # Save and register in Model Registry for Objective 5
    mlflow.sklearn.log_model(final_model, name="best_model", registered_model_name="ChurnPredictor")

    print(f"Best model - accuracy={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")

print("Hyperparameter tuning complete.")