import os, glob, math
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.utils import load_artifact

APP_TITLE = "Sentiment API (TF-IDF + Linear)"
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

def load_best_model(model_dir=MODEL_DIR):
    best = os.path.join(model_dir, "model_best.joblib")
    if os.path.exists(best):
        return load_artifact(best)
    gl = glob.glob(os.path.join(model_dir, "model_*.joblib"))
    assert gl, "No saved model found in models/"
    return load_artifact(gl[0])

def p_positive(model, texts: List[str]):
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        import numpy as np
        return model.predict_proba(texts)[:, 1]
    if hasattr(clf, "decision_function"):
        import numpy as np
        m = model.decision_function(texts)
        return 1/(1+np.exp(-m))
    return None

app = FastAPI(title=APP_TITLE)

class PredictIn(BaseModel):
    text: str
    threshold: Optional[float] = None

class PredictOut(BaseModel):
    label: str
    p_positive: Optional[float] = None
    p_negative: Optional[float] = None
    applied_threshold: Optional[float] = None

@app.on_event("startup")
def _startup():
    global MODEL, LE, THR
    MODEL = load_best_model(MODEL_DIR)
    LE = load_artifact(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    THR = None
    thr_path = os.path.join(MODEL_DIR, "threshold.txt")
    if os.path.exists(thr_path):
        try:
            THR = float(open(thr_path).read().strip())
        except Exception:
            THR = None

@app.post("/predict", response_model=PredictOut)
def predict_one(payload: PredictIn):
    txt = payload.text.strip()
    if not txt:
        return PredictOut(label="unknown")
    p = p_positive(MODEL, [txt])
    if p is None:
        idx = MODEL.predict([txt])[0]
        label = LE.inverse_transform([idx])[0]
        return PredictOut(label=label)
    p = float(p[0])
    thr = payload.threshold if payload.threshold is not None else (THR if THR is not None else 0.5)
    label = "positive" if p >= thr else "negative"
    return PredictOut(label=label, p_positive=p, p_negative=1-p, applied_threshold=thr)

class PredictBatchIn(BaseModel):
    texts: List[str]
    threshold: Optional[float] = None

@app.post("/predict-batch")
def predict_batch(payload: PredictBatchIn):
    texts = [t.strip() for t in payload.texts]
    p = p_positive(MODEL, texts)
    thr = payload.threshold if payload.threshold is not None else (THR if THR is not None else 0.5)
    if p is None:
        idx = MODEL.predict(texts)
        labels = LE.inverse_transform(idx).tolist()
        return {"pred_label": labels}
    p = p.tolist()
    labels = ["positive" if v >= thr else "negative" for v in p]
    return {"pred_label": labels, "p_positive": p, "p_negative": [1-v for v in p], "applied_threshold": thr}
