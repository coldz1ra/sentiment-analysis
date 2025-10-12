import argparse
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.preprocessing import LabelEncoder
from src.preprocess import clean_text
from src.utils import ensure_dir, save_artifact
from src.config import cfg


def build_candidates():
    vec = TfidfVectorizer(preprocessor=clean_text)
    pipe = Pipeline([("tfidf", vec), ("clf", LogisticRegression(max_iter=500))])
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 3, 5],
        "tfidf__max_df": [0.9, 0.95],
        "clf": [LogisticRegression(max_iter=500, class_weight="balanced"),
                LinearSVC()],
        "clf__C": [0.5, 1.0, 2.0]
    }
    return pipe, grid


def main(args):
    df = pd.read_csv(args.data_path)
    assert cfg.text_col in df.columns and cfg.label_col in df.columns
    X = df[cfg.text_col].astype(str).fillna("")
    y_raw = df[cfg.label_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    pipe, grid = build_candidates()
    scorer = make_scorer(f1_score, average="macro")
    gs = GridSearchCV(pipe, grid, scoring=scorer, cv=3, n_jobs=-1, verbose=1)
    gs.fit(Xtr, ytr)

    ensure_dir(args.model_dir)
    best = gs.best_estimator_
    save_artifact(best, os.path.join(args.model_dir, "model_best.joblib"))
    save_artifact(le, os.path.join(args.model_dir, "label_encoder.joblib"))

    y_pred = best.predict(Xte)
    report = classification_report(yte, y_pred, target_names=le.classes_, output_dict=True)
    with open(os.path.join(args.model_dir, "metrics_best.json"), "w") as f:
        json.dump({"best_params": gs.best_params_,
                  "cv_best_score": gs.best_score_, "holdout": report}, f, indent=2)

    import joblib
    joblib.dump({'X_test': Xte.tolist(), 'y_test': yte.tolist()},
                os.path.join(args.model_dir, 'holdout.joblib'))
    print("Best params:", gs.best_params_)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/reviews_mapped.csv")
    parser.add_argument("--model_dir", type=str, default=cfg.model_dir)
    parser.add_argument("--test_size", type=float, default=cfg.test_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()
    main(args)
