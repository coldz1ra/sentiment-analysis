import os
import pandas as pd


def test_mapping_columns_exist():
    p = "data/reviews_mapped.csv"
    assert os.path.exists(p), "run `make map` first"
    df = pd.read_csv(p, nrows=5)
    assert {"text", "label"}.issubset(df.columns)
