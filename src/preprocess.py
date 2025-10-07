from typing import Iterable
import re
import nltk
from nltk.corpus import stopwords

# Ensure resources are present (safe if already downloaded)
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    _ = stopwords.words('english')

STOPWORDS = set(stopwords.words('english'))

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s

def remove_stopwords(tokens: Iterable[str]) -> Iterable[str]:
    return [t for t in tokens if t not in STOPWORDS]
