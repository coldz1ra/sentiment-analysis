import os
import json
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)
from src.utils import load_artifact


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main(args):
    ensure_dir(args.out_dir)
    le = load_artifact(os.path.join(args.model_dir, "label_encoder.joblib"))
    holdout = joblib.load(os.path.join(args.model_dir, "holdout.joblib"))
    X_test = holdout["X_test"]
    y_test = np.array(holdout["y_test"])
    best_path = os.path.join(args.model_dir, "model_best.joblib")
    if os.path.exists(best_path):
        model_path = best_path
    else:
        model_glob = glob.glob(os.path.join(args.model_dir, "model_*.joblib"))
        assert model_glob, "No saved model found in models/"
        model_path = model_glob[0]
    model = load_artifact(model_path)
    print(f"Loaded model: {os.path.basename(model_path)}")
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    with open(os.path.join(args.out_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    fig = plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)
    fig = plt.figure()
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred, normalize="true"),
        display_labels=le.classes_,
    )
    disp.plot(values_format=".2f")
    fig.savefig(
        os.path.join(args.out_dir, "confusion_matrix_norm.png"), bbox_inches="tight"
    )
    plt.close(fig)
    scores = None
    if hasattr(model.named_steps["clf"], "predict_proba") and len(le.classes_) == 2:
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model.named_steps["clf"], "decision_function") and len(le.classes_) == 2:
        scores = model.decision_function(X_test)
    if scores is not None:
        fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
        auc = roc_auc_score(y_test, scores)
        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC (AUC={auc:.3f})")
        fig.savefig(os.path.join(args.out_dir, "roc_curve.png"), bbox_inches="tight")
        plt.close(fig)
        precision, recall, _ = precision_recall_curve(y_test, scores, pos_label=1)
        ap = average_precision_score(y_test, scores)
        fig = plt.figure()
        plt.step(recall, precision, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR curve (AP={ap:.3f})")
        fig.savefig(os.path.join(args.out_dir, "pr_curve.png"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--out_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args)
