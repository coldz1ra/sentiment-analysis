from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def _plot_cm(cm, labels, path: Path, title: str, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    vmax = float(cm.max()) if cm.size and cm.max() > 0 else 1.0
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_title(title)
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([x - 0.5 for x in range(0, n + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(0, n + 1)], minor=True)
    ax.grid(which="minor", color="#999999", linestyle="-", linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if fmt != "d" else str(int(val))
            color = "white" if vmax and val > 0.6 * vmax else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _align_union(y_true, y_pred):
    y_true_s = np.array([str(v) for v in y_true])
    y_pred_s = np.array([str(v) for v in y_pred])
    labels = sorted(np.unique(np.concatenate([y_true_s, y_pred_s])))
    lab2idx = {lab: i for i, lab in enumerate(labels)}
    y_true_idx = np.array([lab2idx[v] for v in y_true_s])
    y_pred_idx = np.array([lab2idx[v] for v in y_pred_s])
    return y_true_s, y_pred_s, y_true_idx, y_pred_idx, labels


def _scores_binary(model, X):
    try:
        proba = model.predict_proba(X)
        if proba is not None:
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, -1]
            if proba.ndim == 1:
                return proba
    except Exception:
        pass
    try:
        dec = model.decision_function(X)
        if dec is not None:
            z = np.array(dec, dtype=float)
            if z.ndim > 1:
                z = z[:, -1]
            z = np.clip(z, -20, 20)
            return 1.0 / (1.0 + np.exp(-z))
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/reviews_mapped.csv")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.model_dir)

    best = mdir / "model_best.joblib"
    model_path = best if best.exists() else sorted(mdir.glob("model_*.joblib"))[0]
    model = load(model_path)
    print(f"Loaded model: {model_path.name}")

    df = pd.read_csv(args.data_path)
    assert {"text", "label"}.issubset(df.columns), "CSV must have 'text' and 'label'"

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    y_pred = model.predict(X_te)
    y_true_s, y_pred_s, y_true_idx, y_pred_idx, labels = _align_union(y_te, y_pred)

    report = classification_report(
        y_true_s, y_pred_s, labels=labels, output_dict=True, zero_division=0
    )
    (out / "classification_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(labels)))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "confusion_matrix.csv")
    _plot_cm(cm, labels, out / "confusion_matrix.png", "Confusion Matrix", "d")

    cmn = cm.astype(float)
    row_sum = cmn.sum(axis=1, keepdims=True)
    cmn = np.divide(cmn, row_sum, out=np.zeros_like(cmn), where=row_sum != 0)
    _plot_cm(cmn, labels, out / "confusion_matrix_norm.png", "Confusion Matrix (normalized)", ".2f")

    scores = _scores_binary(model, X_te)
    if scores is not None and len(labels) == 2:
        pos_label = labels[-1]
        y_bin = (np.array(y_true_s) == pos_label).astype(int)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        RocCurveDisplay.from_predictions(y_bin, scores, ax=ax)
        ax.set_title("ROC Curve")
        fig.tight_layout()
        fig.savefig(out / "roc_curve.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        PrecisionRecallDisplay.from_predictions(y_bin, scores, ax=ax)
        ax.set_title("Precision-Recall Curve")
        fig.tight_layout()
        fig.savefig(out / "pr_curve.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        Path(out / "roc_curve.png").write_bytes(b"")
        Path(out / "pr_curve.png").write_bytes(b"")

    print("CM labels:", labels)
    print("CM matrix:\n", cm)
    print("Saved:", [str(p) for p in [out / "confusion_matrix.png", out /
          "confusion_matrix_norm.png", out / "roc_curve.png", out / "pr_curve.png"]])


if __name__ == "__main__":
    main()
