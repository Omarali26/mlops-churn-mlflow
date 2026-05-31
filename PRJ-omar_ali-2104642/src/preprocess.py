"""
preprocess.py
--------------------
Loads and prepares the Telco Customer Churn dataset for modelling.

Steps:
    1. Drop the customerID column (not a feature).
    2. Label-encode all Yes/No and categorical text columns to integers.
    3. Split into train (80%) and test (20%) sets.

Returns X_train, X_test, y_train, y_test as numpy arrays.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path: str):
    """Return train/test splits ready for sklearn models."""
    df = pd.read_csv(path)

    # Drop identifier column - not useful as a feature
    df.drop(columns=["customerID"], inplace=True)

    # Encode every object (text) column, including the target "Churn"
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # Seperate features (X) from the label (y)
    X = df.drop(columns=["Churn"]).values
    y = df["Churn"].values

    # 80 / 20 stratified split so both sets keep the same churn ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_and_preprocess("data\\telco-customer-churn.csv")