# ADR 0001: Model choice and features

Context:
- Baseline needed for fast CPU inference and reproducible CI.
- Compared Logistic Regression vs Linear SVM on 1–2 n-grams.

Decision:
- Use TF-IDF(1–2) + Logistic Regression, class_weight=balanced.
- Keep optional probability calibration (Platt/Isotonic) for analysis.

Consequences:
- Simple to debug, exportable with joblib, tiny memory footprint.
- Struggles with sarcasm/short texts; mitigated by threshold tuning.
