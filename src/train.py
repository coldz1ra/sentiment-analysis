from sklearn.metrics import classification_report
from src.utils import save_artifact
import os
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def _calibrator(base_clf, method):
    try:
        # sklearn >= 1.4
        return CalibratedClassifierCV(estimator=base_clf, cv=5, method=method)
    except TypeError:
        # sklearn <= 1.3
        return CalibratedClassifierCV(base_estimator=base_clf, cv=5, method=method)


def build_pipeline(model_name: str,
                   ngram_max: int,
                   class_weight_opt: str,
                   stopwords_path: str,
                   calibration: str = "none") -> Pipeline:
    cw = None if str(class_weight_opt).lower() == "none" else class_weight_opt

    stopw = None
    if stopwords_path:
        p = Path(stopwords_path)
        if p.exists():
            stopw = [w.strip() for w in p.read_text().splitlines() if w.strip()]

    tfidf = TfidfVectorizer(ngram_range=(1, ngram_max), stop_words=stopw)

    if model_name == "logreg":
        base_clf = LogisticRegression(max_iter=300, class_weight=cw)
    elif model_name == "svm":
        base_clf = LinearSVC(class_weight=cw)
    else:
        raise ValueError("Unknown --model. Use 'logreg' or 'svm'.")

    clf = base_clf
    if calibration and calibration.lower() != "none":
        method = "isotonic" if calibration.lower() == "isotonic" else "sigmoid"
        clf = _calibrator(base_clf, method)

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def main(args):
    os.makedirs(args.model_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    assert "text" in df.columns and "label" in df.columns, "CSV must have columns 'text' and 'label'."

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    pipe = build_pipeline(args.model, args.ngram_max, args.class_weight,
                          args.stopwords_path, args.calibration)
    pipe.fit(X_train, y_train_enc)

    joblib.dump({"X_test": X_test, "y_test": y_test_enc},
                os.path.join(args.model_dir, "holdout.joblib"))
    save_artifact(le, os.path.join(args.model_dir, "label_encoder.joblib"))

    model_fname = f"model_{args.model}.joblib"
    save_artifact(pipe, os.path.join(args.model_dir, model_fname))

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test_enc, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True)

    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.model_dir, "metrics_best.json"), "w") as f:
        json.dump({"holdout": report}, f, indent=2)

    print(f"Saved model: {model_fname}")
    print("Holdout metrics saved to models/metrics*.json")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="logreg")  # 'logreg' or 'svm'
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--class_weight", type=str, default="balanced")  # 'balanced' or 'None'
    p.add_argument("--stopwords_path", type=str, default="")
    p.add_argument("--calibration", type=str, default="none")  # 'none' | 'sigmoid' | 'isotonic'
    args = p.parse_args()
    main(args)
