"""
main.py
----------
Runs all five objectives in order.

Usage:
    python main.py

This is the single entry-point to reproduce the entire project.
Each objective script is independent and can also be run on its own.
"""
import mlflow
import subprocess
import sys

mlflow.set_tracking_uri("sqlite:///mlflow.db")

scripts = [
    ("Objective 1 - Experiment Tracking",       "src/01_tracking.py"),
    ("Objective 2 - Hyperparameter Tuning",     "src/02_training_tuning.py"),
    ("Objective 3 - Model Deployment",          "src/03_deploy.py"),
    ("Objective 4 - Performance Monitoring",    "src/04_monitor.py"),
    ("Objective 5 - Model Registry",            "src/05_model_registry.py"),
]

for title, script in scripts:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    result = subprocess.run([sys.executable, script], capture_output=False)

    if result.returncode != 0:
        print(f"\n{script} failed with code {result.returncode}. Stoped")
        sys.exit(result.returncode)

print("\n" + "=" * 60)
print("All objectives complete.")
print("=" * 60)