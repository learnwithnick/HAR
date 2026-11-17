"""
Human Activity Recognition (UCI HAR smartphone dataset) - training pipeline.

What it does
  1. Loads the official train / test split (30 subjects: 21 train, 9 test).
  2. Picks the best model using 5-fold GROUPED cross-validation on the
     training subjects (GroupKFold by subject -> no person leaks across folds).
  3. Refits the winner on all training data and evaluates ONCE on the
     untouched test subjects.
  4. Saves the model, metrics, and a confusion-matrix plot.

Run:  python train.py --data "data/UCI HAR Dataset" --out .
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


def load_split(root: Path, split: str):
    """Return features X (n x 561), labels y (1..6) and subject ids for a split."""
    folder = root / split
    X = pd.read_csv(folder / f"X_{split}.txt", sep=r"\s+", header=None).values
    y = pd.read_csv(folder / f"y_{split}.txt", header=None).values.ravel()
    subjects = pd.read_csv(folder / f"subject_{split}.txt", header=None).values.ravel()
    return X, y, subjects


def load_activity_names(root: Path):
    """activity_labels.txt maps 1..6 -> WALKING, WALKING_UPSTAIRS, ..."""
    table = pd.read_csv(root / "activity_labels.txt", sep=r"\s+", header=None, index_col=0)
    return [table.loc[i, 1] for i in sorted(table.index)]


def candidate_models():
    """Models compared with cross-validation. Scaling is inside each pipeline,
    so it is fitted only on the training folds (no leakage)."""
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)
        ),
        "SVM_RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/UCI HAR Dataset")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    root, out = Path(args.data), Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "outputs").mkdir(parents=True, exist_ok=True)

    class_names = load_activity_names(root)
    X_train, y_train, subj_train = load_split(root, "train")
    X_test, y_test, subj_test = load_split(root, "test")

    # Sanity checks: the subjects must not overlap between splits.
    assert set(subj_train).isdisjoint(set(subj_test)), "subject leakage between splits!"
    print(f"Train: {X_train.shape} from {len(set(subj_train))} subjects")
    print(f"Test : {X_test.shape} from {len(set(subj_test))} subjects")
    print("Classes:", class_names)

    # ---- 1. Model selection with subject-grouped CV (test set NOT used here) ----
    cv = GroupKFold(n_splits=5)
    cv_results = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(
            model, X_train, y_train, groups=subj_train, cv=cv, scoring="accuracy"
        )
        cv_results[name] = {
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
        }
        print(f"[CV] {name:20s} accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["cv_accuracy_mean"])
    print(f"\nBest model by CV: {best_name}")

    # ---- 2. Refit the winner on ALL training data ----
    best = candidate_models()[best_name]
    best.fit(X_train, y_train)

    # ---- 3. One-time evaluation on the held-out test subjects ----
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, digits=4
    )
    print(f"\nTEST accuracy = {acc:.4f}   macro-F1 = {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    # ---- 4. Save artifacts ----
    joblib.dump(
        {"model": best, "labels": class_names, "model_name": best_name},
        out / "models" / "har_model.joblib",
    )

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix - {best_name} (test subjects)")
    fig.tight_layout()
    fig.savefig(out / "outputs" / "confusion_matrix.png", dpi=130)

    metrics = {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "per_class": {k: v for k, v in report.items() if k in class_names},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    (out / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model -> models/har_model.joblib")
    print("Saved metrics -> outputs/metrics.json, outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
