import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build(model_name, ngram_max, class_weight, stopwords=None):
    tfidf = TfidfVectorizer(ngram_range=(1, ngram_max), stop_words=stopwords)
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=300,
                                 class_weight=None if class_weight == "None" else class_weight)
    elif model_name == "svm":
        clf = LinearSVC(class_weight=None if class_weight == "None" else class_weight)
    else:
        raise ValueError("use --model logreg|svm")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def load_stopwords(path):
    if not path or not os.path.exists(path):
        return None
    return sorted({w.strip() for w in open(path) if w.strip()})


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.data_path)
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values
    classes = sorted(np.unique(y))

    stopw = load_stopwords(args.stopwords_path)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    rows = []
    for i, (tr, te) in enumerate(cv.split(X, y), 1):
        pipe = build(args.model, args.ngram_max, args.class_weight, stopw)
        pipe.fit(X[tr], y[tr])
        y_pred = pipe.predict(X[te])
        rep = classification_report(y[te], y_pred, target_names=classes, output_dict=True)
        row = {"fold": i,
               "macro_f1": rep["macro avg"]["f1-score"],
               **{f"{c}_f1": rep[c]["f1-score"] for c in classes}}
        rows.append(row)
        print(f"Fold {i}: macro_f1={row['macro_f1']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "cv_metrics.csv"), index=False)
    summary = table.mean(numeric_only=True).to_dict()
    with open(os.path.join(args.out_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Mean macro_f1:", round(summary["macro_f1"], 3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--model", default="logreg")
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--class_weight", default="balanced")
    p.add_argument("--stopwords_path", default="data/domain_stopwords.txt")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    main(args)
