import argparse
import json
from pathlib import Path

import pandas as pd
import mlflow.pyfunc
import mlflow


def _ensure_mlflow_run() -> bool:
    try:
        if mlflow.active_run() is None:
            mlflow.start_run()
        return True
    except Exception as e:
        print(f"[WARN] MLflow logging disabled: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to MLflow model dir (Model asset mounted input)")
    ap.add_argument("--data", type=str, required=True, help="CSV (uri_file asset)")
    ap.add_argument("--out", type=str, required=True, help="Output folder (managed output)")
    ap.add_argument("--rows", type=int, default=200, help="How many rows to score (head)")
    args = ap.parse_args()

    model = mlflow.pyfunc.load_model(args.model)
    df = pd.read_csv(args.data)

    # match train cleaning
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    if not set(cols).issubset(df.columns):
        raise ValueError(f"CSV must contain columns {cols}")

    sample = df.head(args.rows).copy()
    preds = model.predict(sample[cols])
    try:
        sample["prediction"] = pd.Series(preds).astype(int)
    except Exception:
        sample["prediction"] = (pd.Series(preds).astype(float) > 0.5).astype(int)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(outdir / "preds.csv", index=False)

    metrics = {}
    if "Survived" in sample.columns:
        metrics["accuracy_head"] = float((sample["prediction"] == sample["Survived"].astype(int)).mean())
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if _ensure_mlflow_run():
        if "accuracy_head" in metrics:
            mlflow.log_metric("accuracy_head", metrics["accuracy_head"])
        preview = sample.head(25)
        preview_path = outdir / "preds_head.csv"
        preview.to_csv(preview_path, index=False)
        mlflow.log_artifact(str(preview_path), artifact_path="artifacts")

    print(f"[INFO] Wrote {outdir/'preds.csv'} and metrics.json")


if __name__ == "__main__":
    main()
