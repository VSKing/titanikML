import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn


def fail(msg: str, code: int = 2):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)


def _to_py(v):
    """Convert numpy/pandas scalars to plain Python for JSON."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.bool_,)):   return bool(v)
    except Exception:
        pass
    if pd.isna(v): return None
    return v


def _ensure_mlflow_run() -> bool:
    """Start an MLflow run if none is active. Returns True if logging is enabled."""
    try:
        if mlflow.active_run() is None:
            mlflow.start_run()
        return True
    except Exception as e:
        print(f"[WARN] MLflow logging disabled: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="CSV path (AML uri_file input)")
    p.add_argument("--model-dir", type=str, required=True, help="Output dir for the MLflow model")
    args = p.parse_args()

    csv = Path(args.data)
    if not csv.exists():
        fail(f"Dataset not found: {csv}")

    df = pd.read_csv(csv)
    required = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"Missing columns: {missing}")

    # Minimal cleaning
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
    y = df["Survived"].astype(int)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Sex","Embarked"]),
        ("num", "passthrough", ["Pclass","Age","SibSp","Parch","Fare"]),
    ])
    model = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, model.predict(Xte)))
    print(f"[METRIC] accuracy={acc:.4f}")

    out = Path(args.model_dir); out.mkdir(parents=True, exist_ok=True)

    # Save MLflow model
    mlflow.sklearn.save_model(sk_model=model, path=str(out))
    print(f"[INFO] Saved MLflow model to: {out}")

    # Log to AML run
    if _ensure_mlflow_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_text(json.dumps({"accuracy": acc}, indent=2), artifact_file="metrics.json")
        sample = X.iloc[[0]]
        payload = {"input_data": {"columns": list(sample.columns),
                   "data": [[_to_py(v) for v in sample.iloc[0].values]]}}
        mlflow.log_text(json.dumps(payload, indent=2), artifact_file="sample_request.json")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
