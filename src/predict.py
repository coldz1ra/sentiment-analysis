import os
import glob
import argparse

from src.utils import load_artifact


def load_best_model(model_dir: str):
    best_path = os.path.join(model_dir, "model_best.joblib")
    if os.path.exists(best_path):
        return load_artifact(best_path)
    gl = glob.glob(os.path.join(model_dir, "model_*.joblib"))
    assert gl, "No saved model found in models/"
    return load_artifact(gl[0])


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for a given text.")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--text", type=str, required=True, help="Input text to classify")
    args = parser.parse_args()

    model = load_best_model(args.model_dir)
    le = load_artifact(os.path.join(args.model_dir, "label_encoder.joblib"))

    text = args.text.strip()
    if not text:
        print("Empty text.")
        return 1

    pred_idx = model.predict([text])[0]
    label = le.inverse_transform([pred_idx])[0]
    thr_file = os.path.join(args.model_dir, 'threshold.txt')
    thr = None
    if os.path.exists(thr_file):
        try:
            thr = float(open(thr_file).read().strip())
        except Exception:
            thr = None
    # если модель умеет вероятности — покажем confidence
    prob = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        prob = float(model.predict_proba([text])[:, 1][0])
    elif hasattr(model.named_steps["clf"], "decision_function"):
        # приведём margin к [0,1] через сигмоиду — ориентировочная уверенность
        import math
        prob = 1 / (1 + math.exp(-float(model.decision_function([text])[0])))

    if prob is not None and thr is not None:
        label_by_thr = 'positive' if prob >= thr else 'negative'
        print(f"Prediction: {label_by_thr} (p_pos={prob:.3f}, thr={thr:.3f})")
    elif prob is not None:
        print(f"Prediction: {label} (p_pos={prob:.3f})")
    else:
        print(f"Prediction: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
