import argparse, os, json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
from src.utils import load_artifact
from src.config import cfg
import joblib

def plot_confusion(cm, classes, out_path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def main(args):
    import glob
    best_path = os.path.join(args.model_dir, "model_best.joblib")
    if os.path.exists(best_path):
        model_path = best_path
    else:
        model_glob = glob.glob(os.path.join(args.model_dir, "model_*.joblib"))
        assert model_glob, "No saved model found in models/"
        model_path = model_glob[0]
    model = load_artifact(model_path)
    print(f"Loaded model: {os.path.basename(model_path)}")

    le = load_artifact(os.path.join(args.model_dir, "label_encoder.joblib"))
    holdout = joblib.load(os.path.join(args.model_dir, 'holdout.joblib'))
    X_test = holdout['X_test']
    y_test = np.array(holdout['y_test'])

    # Predict
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))))
    plot_confusion(cm, le.classes_, os.path.join(args.out_dir, "confusion_matrix.png"))

    # ROC only for binary classifiers with proba
    if hasattr(model.named_steps['clf'], 'predict_proba') and len(le.classes_) == 2:
        fig = plt.figure()
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        fig.savefig(os.path.join(args.out_dir, 'roc_curve.png'), bbox_inches='tight')
        plt.close(fig)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=cfg.model_dir)
    parser.add_argument("--out_dir", type=str, default=cfg.out_dir)
    args = parser.parse_args()
    main(args)
