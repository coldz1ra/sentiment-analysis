import subprocess
import os


def test_train_on_small_sample():
    code = subprocess.call(
        ["bash", "-lc", ". .venv/bin/activate && python -m src.train --data_path data/reviews_mapped.csv --model_dir models --test_size 0.5 --seed 0 --model logreg"])  # noqa: E501
    assert code == 0
    assert os.path.exists("models/label_encoder.joblib")
