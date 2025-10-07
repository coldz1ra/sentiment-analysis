import os, json, glob
import numpy as np
from sklearn.metrics import classification_report
from src.utils import load_artifact
from src.config import cfg
import joblib

MODEL_DIR = cfg.model_dir

# выбираем модель: сперва model_best, иначе любую model_*.joblib
best = os.path.join(MODEL_DIR, "model_best.joblib")
if os.path.exists(best):
    model_path = best
else:
    gl = glob.glob(os.path.join(MODEL_DIR, "model_*.joblib"))
    assert gl, "No model found in models/"
    model_path = gl[0]

model = load_artifact(model_path)
le = load_artifact(os.path.join(MODEL_DIR, "label_encoder.joblib"))
holdout = joblib.load(os.path.join(MODEL_DIR, "holdout.joblib"))
X_test = holdout["X_test"]; y_test = np.array(holdout["y_test"])

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

# атомарная запись файлов метрик
tmp_best = os.path.join(MODEL_DIR, "_metrics_best.tmp.json")
with open(tmp_best, "w") as f:
    json.dump({"holdout": report, "model_file": os.path.basename(model_path)}, f, indent=2)
os.replace(tmp_best, os.path.join(MODEL_DIR, "metrics_best.json"))

tmp_std = os.path.join(MODEL_DIR, "_metrics.tmp.json")
with open(tmp_std, "w") as f:
    json.dump(report, f, indent=2)
os.replace(tmp_std, os.path.join(MODEL_DIR, "metrics.json"))

print("Rebuilt: models/metrics_best.json and models/metrics.json from", os.path.basename(model_path))

