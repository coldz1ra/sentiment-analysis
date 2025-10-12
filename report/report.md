# Sentiment Analysis — Report

**Date:** 2025-10-06

## 1. Objective
Classify review sentiment and extract actionable insights to improve product experience.

## 2. Dataset
- Reviews dataset with columns: `text`, `label` (`positive`, `negative`, optionally `neutral`).
- Balanced classes preferable; otherwise discuss mitigation (class weights / thresholding).

## 3. Methodology
- **Preprocessing:** lowercasing, punctuation/number cleanup, stopword handling via NLTK.
- **Vectorization:** TF-IDF (1–2 grams), min_df=3, max_df=0.9, sublinear_tf.
- **Models:** Logistic Regression (baseline), Linear SVM (alt).
- **Evaluation:** Accuracy, Precision, Recall, F1; Confusion Matrix; ROC-AUC (binary).

## 4. Key Findings (template)
- Positive reviews often mention: *delivery speed*, *quality*, *price/value*.
- Negative reviews often mention: *packaging*, *defects*, *support*, *refunds*.
- Words with strongest positive weight examples: *excellent, love, perfect*.
- Words with strongest negative weight examples: *broken, terrible, refund*.

## 5. Recommendations
- Improve **support SLAs** and **return/refund flow**.
- Prioritize **quality checks** for SKUs with high defect mentions.
- Promote features praised in positive reviews in marketing copy.

## 6. Next Steps
- Add **neutral** class + calibration.
- Try **fastText** or **DistilBERT** for comparison.
- Build a **Streamlit** demo for live predictions.

## 7. Results
- Best config: see `models/metrics_best.json`
- Holdout macro F1: 0.9222
negative: P=0.927 R=0.917 F1=0.922
positive: P=0.918 R=0.928 F1=0.923
- Figures: `outputs/` (confusion matrix, ROC)
