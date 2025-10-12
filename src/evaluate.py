import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import load  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split  # noqa: E402


def _plot_cm(cm, labels, path: Path, title: str, fmt: str) -> None:
    import matplotlib.pyplot as plt
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    vmax = float(cm.max()) if cm.size and cm.max() > 0 else 1.0
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
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
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    vmax = float(cm.max()) if cm.size and cm.max() > 0 else 1.0
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/reviews_mapped.csv")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.model_dir)

    best = mdir / "model_best.joblib"
    if best.exists():
        model_path = best
    else:
        candidates = sorted(mdir.glob("model_*.joblib"))
        assert candidates, "No saved model in models/"
        model_path = candidates[0]
    model = load(model_path)
    print(f"Loaded model: {model_path.name}")

    df = pd.read_csv(args.data_path)
    assert {"text", "label"}.issubset(df.columns), "CSV must have columns 'text' and 'label'"

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    y_pred = model.predict(X_te)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_te)[:, 1]
        except Exception:
            y_proba = None

    report = classification_report(y_te, y_pred, output_dict=True)
    (out / "classification_report.json").write_text(json.dumps(report, indent=2))

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    _plot_cm(cm, labels, out / "confusion_matrix.png", "Confusion Matrix", "d")

    cmn = cm.astype(float)
    row_sum = cmn.sum(axis=1, keepdims=True)
    cmn = np.divide(cmn, row_sum, out=np.zeros_like(cmn), where=row_sum != 0)
    _plot_cm(cmn, labels, out / "confusion_matrix_norm.png", "Confusion Matrix (normalized)", ".2f")

    if y_proba is not None:
        y_bin = (y_te == labels[-1]).astype(int)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        RocCurveDisplay.from_predictions(y_bin, y_proba, ax=ax)
        ax.set_title("ROC Curve")
        fig.tight_layout()
        fig.savefig(out / "roc_curve.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        PrecisionRecallDisplay.from_predictions(y_bin, y_proba, ax=ax)
        ax.set_title("Precision-Recall Curve")
        fig.tight_layout()
        fig.savefig(out / "pr_curve.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
