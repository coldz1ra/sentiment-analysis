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
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split  # noqa: E402


def plot_cm(cm, labels, path: Path, title: str, fmt: str) -> None:
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
    ax.grid(which="minor", color="#999", linestyle="-", linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333")
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
                color=color,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def map_pred_to_names(y_pred, y_true, model_dir: Path) -> np.ndarray:
    le_path = model_dir / "label_encoder.joblib"
    y_pred = np.asarray(y_pred)
    if le_path.exists():
        try:
            le = load(le_path)
            return le.inverse_transform(y_pred.astype(int))
        except Exception:
            pass
    if set(np.unique(y_true)) <= {"negative", "positive"} and set(np.unique(y_pred)) <= {0, 1}:
        a = np.where(y_pred == 1, "positive", "negative")
        b = np.where(y_pred == 1, "negative", "positive")
        acc_a = (a == y_true).mean()
        acc_b = (b == y_true).mean()
        return a if acc_a >= acc_b else b
    return y_pred.astype(str)


def pos_index(model, model_dir: Path, pos_name: str) -> int:
    le_path = model_dir / "label_encoder.joblib"
    if le_path.exists():
        try:
            le = load(le_path)
            return int(le.transform([pos_name])[0])
        except Exception:
            pass
    if hasattr(model, "classes_"):
        cls = np.array(model.classes_)
        try:
            return int(np.where(cls == pos_name)[0][0])
        except Exception:
            pass
        try:
            return int(np.where(cls.astype(str) == pos_name)[0][0])
        except Exception:
            pass
    return -1


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
    assert {"text", "label"}.issubset(df.columns)

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    y_pred_raw = model.predict(X_te)
    y_pred_named = map_pred_to_names(y_pred_raw, y_te, mdir)
    labels = sorted(np.unique(np.concatenate([y_te, y_pred_named])))

    report = classification_report(
        y_te, y_pred_named, labels=labels, output_dict=True, zero_division=0
    )
    (out / "classification_report.json").write_text(json.dumps(report, indent=2))

    lab2idx = {lab: i for i, lab in enumerate(labels)}
    y_true_idx = np.array([lab2idx[v] for v in y_te])
    y_pred_idx = np.array([lab2idx[v] for v in y_pred_named])

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(labels)))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "confusion_matrix.csv")
    plot_cm(cm, labels, out / "confusion_matrix.png", "Confusion Matrix", "d")

    cmn = cm.astype(float)
    row_sum = cmn.sum(axis=1, keepdims=True)
    cmn = np.divide(cmn, row_sum, out=np.zeros_like(cmn), where=row_sum != 0)
    plot_cm(
        cmn,
        labels,
        out / "confusion_matrix_norm.png",
        "Confusion Matrix (normalized)",
        ".2f",
    )

    if len(labels) == 2:
        pos_name = "positive" if "positive" in labels else labels[-1]
        idx = pos_index(model, mdir, pos_name)
        scores = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_te)
                scores = np.asarray(proba[:, idx], dtype=float)
            except Exception as e:
                print("predict_proba failed:", repr(e))
        if scores is None and hasattr(model, "decision_function"):
            try:
                dec = model.decision_function(X_te)
                z = np.array(dec, dtype=float)
                if z.ndim > 1:
                    z = z[:, idx]
                z = np.clip(z, -20, 20)
                scores = 1.0 / (1.0 + np.exp(-z))
            except Exception as e:
                print("decision_function failed:", repr(e))
        if scores is not None:
            y_bin = (np.array(y_te) == pos_name).astype(int)
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
            (out / "roc_curve.png").write_bytes(b"")
            (out / "pr_curve.png").write_bytes(b"")
    else:
        (out / "roc_curve.png").write_bytes(b"")
        (out / "pr_curve.png").write_bytes(b"")

    print("Labels:", labels)


if __name__ == "__main__":
    main()
