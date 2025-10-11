import os, glob
import streamlit as st
from src.utils import load_artifact

def load_best_model(model_dir="models"):
    best = os.path.join(model_dir, "model_best.joblib")
    if os.path.exists(best):
        return load_artifact(best)
    gl = glob.glob(os.path.join(model_dir, "model_*.joblib"))
    assert gl, "No saved model found in models/"
    return load_artifact(gl[0])

st.set_page_config(page_title="Sentiment Baseline", page_icon="🧠", layout="centered")
st.title("🧠 Sentiment Analysis — Baseline (TF-IDF + Linear)")
st.caption("CPU-friendly, interpretable baseline for reviews")

with st.sidebar:
    st.header("Model")
    model_dir = st.text_input("Model dir", value="models")

try:
    model = load_best_model(model_dir)
    le = load_artifact(os.path.join(model_dir, "label_encoder.joblib"))
    thr = None
    thr_path = os.path.join(model_dir, 'threshold.txt')
    if os.path.exists(thr_path):
        try:
            thr = float(open(thr_path).read().strip())
        except Exception:
            thr = None
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

txt = st.text_area("Enter review text", height=150, placeholder="Type here…")
btn = st.button("Predict")

def predict_proba_safe(mdl, text):
    if hasattr(mdl.named_steps["clf"], "predict_proba"):
        return float(mdl.predict_proba([text])[:,1][0])
    elif hasattr(mdl.named_steps["clf"], "decision_function"):
        import math
        return 1/(1+math.exp(-float(mdl.decision_function([text])[0])))
    return None

if btn and txt.strip():
    pred_idx = model.predict([txt])[0]
    label = le.inverse_transform([pred_idx])[0]
    conf = predict_proba_safe(model, txt)
    if conf is not None and thr is not None:
        label_thr = 'positive' if conf >= thr else 'negative'
        st.subheader(f"Prediction: **{label_thr}** (thr={thr:.3f})")
        st.progress(conf)
        st.write(f"p_pos = **{conf:.3f}**")
    else:
        st.subheader(f"Prediction: **{label}**")
        if conf is not None:
            st.progress(conf)
            st.write(f"p_pos = **{conf:.3f}**")

st.divider()
st.caption("Artifacts: confusion matrices, ROC/PR are in `outputs/` after `make evaluate`.")
