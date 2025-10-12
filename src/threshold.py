import os
import glob
import json
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve, f1_score
from src.utils import load_artifact


def load_best_model(model_dir):
    best = os.path.join(model_dir, "model_best.joblib")
    if os.path.exists(best):
        return load_artifact(best)
    gl = glob.glob(os.path.join(model_dir, "model_*.joblib"))
    assert gl, "No saved model found in models/"
    return load_artifact(gl[0])


def get_scores(model, X):
    if hasattr(model.named_steps["clf"], "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model.named_steps["clf"], "decision_function"):
        m = model.decision_function(X)
        return 1 / (1 + np.exp(-m))  # сигмоида для SVM
    raise RuntimeError("Classifier has neither predict_proba nor decision_function")


def main(model_dir, out_file):
    holdout = joblib.load(os.path.join(model_dir, "holdout.joblib"))
    X_test, y_test = holdout["X_test"], np.array(holdout["y_test"])
    model = load_best_model(model_dir)

    scores = get_scores(model, X_test)
    prec, rec, thr = precision_recall_curve(y_test, scores, pos_label=1)

    # найдём порог, дающий лучший F1
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        y_hat = (scores >= t).astype(int)
        f1 = f1_score(y_test, y_hat)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    os.makedirs(model_dir, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(f"{best_t:.3f}\n")
    print(json.dumps({"best_threshold": round(best_t, 3), "best_f1": round(best_f1, 3)}))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--out", type=str, default="models/threshold.txt")
    args = p.parse_args()
    main(args.model_dir, args.out)
