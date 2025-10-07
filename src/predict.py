import argparse, os, json
from src.utils import load_artifact
from src.config import cfg

def main(args):
    model = load_artifact(os.path.join(args.model_dir, 'model_logreg.joblib'))
    le = load_artifact(os.path.join(args.model_dir, 'label_encoder.joblib'))
    text = args.text
    pred_idx = model.predict([text])[0]
label = le.inverse_transform([pred_idx])[0]

if hasattr(model.named_steps['clf'], 'predict_proba'):
    probs = model.predict_proba([text])[0]
    pos_idx = 1 if len(le.classes_) == 2 else probs.argmax()

    tpath = os.path.join(args.model_dir, "threshold.json")
    if os.path.exists(tpath) and len(le.classes_) == 2:
        thr = json.load(open(tpath))["best_threshold"]
        pred_idx = int(probs[pos_idx] >= thr)
        label = le.inverse_transform([pred_idx])[0]

    print(f"Prediction: {label} (confidence={probs[pos_idx]:.3f})")
else:
    print(f"Prediction: {label}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Text to classify')
    parser.add_argument('--model_dir', type=str, default=cfg.model_dir)
    args = parser.parse_args()
    main(args)
