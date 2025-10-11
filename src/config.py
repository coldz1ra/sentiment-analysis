from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    text_col: str = "text"
    label_col: str = "label"
    model_dir: str = "models"
    out_dir: str = "outputs"
    test_size: float = 0.2
    seed: int = 42
    classes: List[str] = None  # optional fixed ordering


cfg = Config()
