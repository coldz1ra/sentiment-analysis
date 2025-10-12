import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='data/reviews_mapped.csv')
    p.add_argument('--model_dir', default='models')
    p.add_argument('--out_dir', default='outputs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test_size', type=float, default=0.2)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.model_dir)
    model_path = (
        mdir /
        'model_best.joblib') if (
        mdir /
        'model_best.joblib').exists() else sorted(
            mdir.glob('model_*.joblib'))[0]
    model = load(model_path)
    print(f'Loaded model: {model_path.name}')

    df = pd.read_csv(args.data_path)
    assert {'text', 'label'}.issubset(df.columns), "CSV must have columns 'text' and 'label'"
    X = df['text'].astype(str).values
    y = df['label'].astype(str).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    y_pred = model.predict(X_te)
    # probas — если есть
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_te)[:, 1]
        except Exception:
            y_proba = None

    # отчёт
    rep = classification_report(y_te, y_pred, output_dict=True)
    (out / 'classification_report.json').write_text(json.dumps(rep, indent=2))

    # Confusion Matrix (count)
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Confusion Matrix (normalized)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    disp = ConfusionMatrixDisplay(cmn, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='.2f')
    ax.set_title('Confusion Matrix (normalized)')
    plt.tight_layout()
    fig.savefig(out / 'confusion_matrix_norm.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ROC / PR (если доступны вероятности)
    if y_proba is not None:
        # ROC
        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        try:
            RocCurveDisplay.from_predictions((y_te == labels[-1]).astype(int), y_proba, ax=ax)
            ax.set_title('ROC Curve')
            plt.tight_layout()
            fig.savefig(out / 'roc_curve.png', bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)
        # PR
        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        try:
            PrecisionRecallDisplay.from_predictions(
                (y_te == labels[-1]).astype(int), y_proba, ax=ax)
            ax.set_title('Precision-Recall Curve')
            plt.tight_layout()
            fig.savefig(out / 'pr_curve.png', bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)


if __name__ == '__main__':
    main()
