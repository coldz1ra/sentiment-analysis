# Model Card
- Task: Binary sentiment classification
- Data: short user reviews (no PII), see DATA.md
- Model: TF-IDF (1–2) + Logistic Regression (class_weight=balanced), optional calibration
- Metrics (holdout): see README
- Limitations: sarcasm, very short inputs, domain shift
- Intended use: demos, learning; not for high-stakes decisions
