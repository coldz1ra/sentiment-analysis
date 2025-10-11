import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config import cfg
from src.utils import ensure_dir, save_artifact
from src.preprocess import clean_text


def build_pipeline(model: str):
    if model == "logreg":
        clf = LogisticRegression(max_iter=300, class_weight="balanced")
    elif model == "linearsvc":
        clf = LinearSVC()
    else:
        raise ValueError("Unknown model: %s" % model)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])
    return pipe


def main(args):
    df = pd.read_csv(args.data_path)
    assert cfg.text_col in df.columns and cfg.label_col in df.columns, f"Dataset must have columns '{cfg.text_col}' and '{cfg.label_col}'."

    X = df[cfg.text_col].astype(str).fillna("")
    y_raw = df[cfg.label_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = build_pipeline(args.model)
    pipe.fit(X_train, y_train)

    ensure_dir(args.model_dir)
    save_artifact(pipe, os.path.join(args.model_dir, f"model_{args.model}.joblib"))
    save_artifact(le, os.path.join(args.model_dir, "label_encoder.joblib"))

    # persist test split and basic metrics
    import joblib
    joblib.dump({'X_test': X_test.tolist(), 'y_test': y_test.tolist()},
                os.path.join(args.model_dir, 'holdout.joblib'))
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=cfg.model_dir)
    parser.add_argument("--test_size", type=float, default=cfg.test_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "linearsvc"])
    args = parser.parse_args()
    main(args)
