import os
import glob
import math
import streamlit as st
from src.utils import load_artifact


def load_best_model(model_dir="models"):
    best = os.path.join(model_dir, "model_best.joblib")
    if os.path.exists(best):
        return load_artifact(best)
    gl = glob.glob(os.path.join(model_dir, "model_*.joblib"))
    assert gl, "No saved model found in models/"
    return load_artifact(gl[0])


def p_positive(model, text):
    if hasattr(model.named_steps["clf"], "predict_proba"):
        return float(model.predict_proba([text])[:, 1][0])
    if hasattr(model.named_steps["clf"], "decision_function"):
        return 1 / (1 + math.exp(-float(model.decision_function([text])[0])))
    return None


st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")
st.title("💬 Sentiment Analysis — Baseline (TF-IDF + Linear)")

with st.sidebar:
    st.header("Model")
    model_dir = st.text_input("Model directory", value="models")

try:
    model = load_best_model(model_dir)
    le = load_artifact(os.path.join(model_dir, "label_encoder.joblib"))
    thr = None
    thr_path = os.path.join(model_dir, "threshold.txt")
    if os.path.exists(thr_path):
        try:
            thr = float(open(thr_path).read().strip())
        except Exception:
            thr = None
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

txt = st.text_area("Enter review text", height=150, placeholder="e.g., Very dirty public restroom")
btn = st.button("Predict")

if btn and txt.strip():
    p_pos = p_positive(model, txt)
    if p_pos is None:
        pred_idx = model.predict([txt])[0]
        label = le.inverse_transform([pred_idx])[0]
        st.subheader(f"Prediction: **{label.capitalize()}**")
    else:
        p_neg = 1 - p_pos
        applied_thr = 0.5 if thr is None else thr
        label_by_thr = "Positive" if p_pos >= applied_thr else "Negative"
        st.subheader(f"Prediction: **{label_by_thr}**  (threshold={applied_thr:.2f})")
        c1, c2 = st.columns(2)
        c1.metric("Prob. Positive", f"{p_pos*100:.1f}%")
        c2.metric("Prob. Negative", f"{p_neg*100:.1f}%")
        st.progress(p_pos)
        st.caption("Progress bar shows p(Positive).")

st.divider()
st.caption("Quality artifacts live in `outputs/` after running `make evaluate`.")
