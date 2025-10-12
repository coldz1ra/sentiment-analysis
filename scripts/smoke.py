# scripts/smoke.py
import os
import sys
import json
import importlib
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)  # project root
sys.path.insert(0, ROOT)

print("== Smoke check ==")

# Python / venv
print("Python:", sys.executable)

# deps
for mod in [
    "pandas",
    "numpy",
    "sklearn",
    "nltk",
    "matplotlib",
    "wordcloud",
    "joblib",
    "tqdm",
        "streamlit"]:
    try:
        importlib.import_module(mod)
    except Exception as e:
        print(f"[FAIL] import {mod}: {e}")
        sys.exit(1)

# data
data_candidates = [
    os.path.join(ROOT, "data", "reviews_mapped.csv"),
    os.path.join(ROOT, "data", "reviews.csv"),
]
data_path = next((p for p in data_candidates if os.path.exists(p)), None)
print("Data file:", data_path or "NOT FOUND")

if data_path:
    df = pd.read_csv(data_path, nrows=5)
    print("Columns:", list(df.columns))
    need = {"text", "label"} if data_path.endswith("reviews_mapped.csv") else set()
    if need and not need.issubset(df.columns):
        print("[FAIL] expected columns text,label in reviews_mapped.csv")
        sys.exit(1)

# models
md = os.path.join(ROOT, "models")
os.makedirs(md, exist_ok=True)
print("Models dir:", md, "files:", os.listdir(md))

print("[OK] smoke done")
