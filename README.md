# Sentiment Analysis of Reviews

A clean, production-like **NLP project** that classifies review sentiment (positive / negative / neutral), explores the data, and reports business insights.

## ✨ Highlights
- Reproducible pipeline (`src/`) with **TF-IDF + Logistic Regression** baseline and optional **Linear SVM**.
- Clean EDA with charts and **word clouds** for positive/negative classes.
- Robust evaluation with **precision / recall / F1**, **confusion matrix**, and **ROC-AUC**.
- Clear **report** with findings and business recommendations.
- Easy to run: `make setup && make train && make evaluate`.

## 📂 Structure
```
sentiment-analysis/
├─ data/               # Put your dataset CSV here (e.g., IMDB 50K, Amazon Reviews subset)
├─ models/             # Saved models, vectorizers, and label encoders
├─ notebooks/
│  └─ sentiment_analysis.ipynb
├─ outputs/            # Figures, reports exports
├─ report/
│  └─ report.md
├─ src/
│  ├─ config.py
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ utils.py
├─ requirements.txt
├─ Makefile
└─ README.md
```

## 🗃️ Data
Use any review dataset with at least two columns:
- `text` — raw review text
- `label` — sentiment class (`positive`, `negative`, optional `neutral`)

**Recommended**: IMDB 50K movie reviews (CSV) or Amazon reviews subset.  
Place the CSV as `data/reviews.csv`. If your dataset has different column names, adjust in `src/config.py`.

## ⚙️ Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
```

## ▶️ Quickstart
```bash
# 1) Put your CSV to data/reviews.csv
make train          # trains TF-IDF + Logistic Regression
make evaluate       # prints metrics & saves confusion matrix/ROC
```

## 🔧 Make targets
```bash
make setup          # create venv and install requirements
make train          # run training
make evaluate       # evaluate on holdout
make clean          # remove models/ and outputs/
```

## 📑 Reporting
Main analytical report lives in `report/report.md`. Exported figures render to `outputs/`.

## 🧪 Notes
- No internet required for training. Only NLTK tokenizers/stopwords download once.
- Works with **CPU** only.
- Python 3.9+.


![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)
![License](https://img.shields.io/badge/license-MIT-informational)

## 🧪 Testing & Linting
```bash
pytest -q
flake8 src
```

## 🌐 Streamlit demo
```bash
streamlit run app/app.py
```

## 🔮 Predict from CLI
```bash
python src/predict.py --text "I absolutely love this product!"
```
