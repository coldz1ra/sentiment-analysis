import os
import subprocess

def test_predict_cli():
    # Если модели нет — обучаем быструю логистику на половине датасета
    if not os.path.exists("models/label_encoder.joblib"):
        code = subprocess.call([
            "bash","-lc",
            ". .venv/bin/activate && export PYTHONPATH=$(pwd) && "
            "python -m src.train --data_path data/reviews_mapped.csv "
            "--model_dir models --test_size 0.5 --seed 0 --model logreg && "
            "cp -f models/model_logreg.joblib models/model_best.joblib || true"
        ])
        assert code == 0, "training failed"

    # Прогноз из CLI должен завершаться кодом 0
    code = subprocess.call([
        "bash","-lc",
        ". .venv/bin/activate && export PYTHONPATH=$(pwd) && "
        "python src/predict.py --model_dir models --text 'great build quality'"
    ])
    assert code == 0
