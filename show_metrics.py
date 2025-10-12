import os
import json
p_best = "models/metrics_best.json"
p_std = "models/metrics.json"
path = p_best if os.path.exists(p_best) else p_std
print("Reading:", path)
with open(path, "r") as f:
    data = json.load(f)

report = data.get("holdout", data)  # у tune есть "holdout", у train — сразу отчёт
macro = report["macro avg"]["f1-score"]
print("Macro F1:", round(macro, 4))
for cls in [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]:
    r = report[cls]
    print(f"{cls}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")
