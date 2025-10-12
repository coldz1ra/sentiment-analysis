from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from joblib import load
from pathlib import Path as _Path

app = FastAPI()

MODEL = None
MODEL_PATH = _Path('models/model_best.joblib')
THRESH_PATH = _Path('models/threshold.txt')


def _load_model():
    global MODEL
    if MODEL is None and MODEL_PATH.exists():
        MODEL = load(MODEL_PATH)
    thr = 0.5
    if THRESH_PATH.exists():
        try:
            thr = float(THRESH_PATH.read_text().strip())
        except Exception:
            thr = 0.5
    return MODEL, thr


class PredictIn(BaseModel):
    text: str
    threshold: Optional[float] = None


class PredictBatchIn(BaseModel):
    texts: List[str]
    threshold: Optional[float] = None


class PredictOut(BaseModel):
    label: str
    confidence: float


@app.get('/health')
def health():
    return {'status': 'ok'}


def _probs(model, texts: List[str]):
    return model.predict_proba(texts)[:, 1]


@app.on_event('startup')
def on_startup():
    _load_model()


@app.post('/predict', response_model=PredictOut)
def predict_one(payload: PredictIn):
    model, thr = _load_model()
    if model is None:
        return PredictOut(label='unknown', confidence=0.0)
    t = payload.threshold if payload.threshold is not None else thr
    txt = payload.text.strip()
    if not txt:
        return PredictOut(label='unknown', confidence=0.0)
    p = float(_probs(model, [txt])[0])
    label = 'positive' if p >= t else 'negative'
    conf = p if label == 'positive' else 1.0 - p
    return PredictOut(label=label, confidence=float(conf))


@app.post('/predict-batch')
def predict_batch(payload: PredictBatchIn):
    model, thr = _load_model()
    if model is None:
        return []
    t = payload.threshold if payload.threshold is not None else thr
    texts = [x.strip() for x in payload.texts]
    ps = _probs(model, texts)
    out = []
    for p in ps:
        p = float(p)
        label = 'positive' if p >= t else 'negative'
        conf = p if label == 'positive' else 1.0 - p
        out.append({'label': label, 'confidence': float(conf)})
    return out
