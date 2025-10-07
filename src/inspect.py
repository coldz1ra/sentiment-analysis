import os, glob, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from src.utils import load_artifact
from src.config import cfg
import joblib

MODEL_DIR = cfg.model_dir
OUT_DIR = cfg.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# 1) выбираем модель: сперва model_best, иначе любую model_*.joblib
best_path = os.path.join(MODEL_DIR, "model_best.joblib")
if os.path.exists(best_path):
    model_path = best_path
else:
    gl = glob.glob(os.path.join(MODEL_DIR, "model_*.joblib"))
    assert gl, "No model found in models/ (ожидаю model_best.joblib или model_*.joblib)"
    model_path = gl[0]

model = load_artifact(model_path)
le = load_artifact(os.path.join(MODEL_DIR, "label_encoder.joblib"))
holdout = joblib.load(os.path.join(MODEL_DIR, "holdout.joblib"))
X_test = holdout["X_test"]; y_test = np.array(holdout["y_test"])

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

# 3) топ-слова по весам (модель линейная с coef_)
clf = model.named_steps["clf"]
if hasattr(clf, "coef_"):
    vec = model.named_steps["tfidf"]
    import numpy as np
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_
    classes = list(le.classes_)
    topn = 30

    weights_by_class = {}

    if coefs.shape[0] == 1 and len(classes) == 2:
        # бинарный случай: coef_ — один вектор для класса 1; для класса 0 — его отрицание
        w = coefs[0]
        # вектор направлен на «второй» класс по индексу 1
        weights_by_class[classes[1]] = w
        weights_by_class[classes[0]] = -w
    else:
        # многоклассовый случай: по одному вектору на класс
        for i, cls in enumerate(classes):
            weights_by_class[cls] = coefs[i]

    # сохранить топ-слова для каждого класса
    for cls, w in weights_by_class.items():
        idx = np.argsort(w)[-topn:][::-1]
        out_df = pd.DataFrame({"feature": feature_names[idx], "weight": w[idx]})
        out_df.to_csv(os.path.join(OUT_DIR, f"top_words_{cls}.csv"), index=False)
else:
    # для моделей без coef_ — пропустить топ-слова
    open(os.path.join(OUT_DIR, "top_words_skipped.txt"), "w").write(
        "Top words skipped: classifier has no coef_ (e.g., non-linear or incompatible)."
    )

