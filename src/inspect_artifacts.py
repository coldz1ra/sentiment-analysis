import os, glob, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_artifact
from src.config import cfg

MODEL_DIR = cfg.model_dir
OUT_DIR = cfg.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# 1) выбрать модель: сначала model_best, иначе первая model_*.joblib
best_path = os.path.join(MODEL_DIR, "model_best.joblib")
if os.path.exists(best_path):
    model_path = best_path
else:
    gl = glob.glob(os.path.join(MODEL_DIR, "model_*.joblib"))
    assert gl, "No model found in models/"
    model_path = gl[0]

model = load_artifact(model_path)
le = load_artifact(os.path.join(MODEL_DIR, "label_encoder.joblib"))
holdout = joblib.load(os.path.join(MODEL_DIR, "holdout.joblib"))
X_test = holdout["X_test"]
y_test = np.array(holdout["y_test"])

# 2) ошибки (FP/FN)
y_pred = model.predict(X_test)
errors = []
for x, yt, yp in zip(X_test, y_test, y_pred):
    if yt != yp:
        errors.append({
            "text": x,
            "true": le.inverse_transform([yt])[0],
            "pred": le.inverse_transform([yp])[0]
        })
pd.DataFrame(errors).to_csv(os.path.join(OUT_DIR, "errors.csv"), index=False)

# 3) топ-слова по весам
clf = model.named_steps["clf"]
if hasattr(clf, "coef_"):
    vec = model.named_steps["tfidf"]
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_
    classes = list(le.classes_)
    topn = 30

    weights_by_class = {}
    if coefs.shape[0] == 1 and len(classes) == 2:
        w = coefs[0]
        # считаем, что положительный класс соответствует classes[1]
        weights_by_class[classes[1]] = w
        weights_by_class[classes[0]] = -w
    else:
        for i, cls in enumerate(classes):
            weights_by_class[cls] = coefs[i]

    for cls, w in weights_by_class.items():
        idx = np.argsort(w)[-topn:][::-1]
        dfw = pd.DataFrame({"feature": feature_names[idx], "weight": w[idx]})
        dfw.to_csv(os.path.join(OUT_DIR, f"top_words_{cls}.csv"), index=False)

        # и PNG-график
        fig = plt.figure(figsize=(8,6))
        plt.barh(dfw["feature"].iloc[::-1], dfw["weight"].iloc[::-1])
        plt.title(f"Top words — {cls}")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"top_words_{cls}.png"), bbox_inches="tight")
        plt.close(fig)
else:
    open(os.path.join(OUT_DIR, "top_words_skipped.txt"), "w").write(
        "Classifier has no coef_; top words skipped."
    )

print("Saved into outputs/:", [f for f in os.listdir(OUT_DIR) if f.startswith(("errors","top_words","confusion","roc","pr_"))])
