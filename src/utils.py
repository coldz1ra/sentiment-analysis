import os
import joblib
from typing import Any


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_artifact(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def load_artifact(path: str):
    return joblib.load(path)
