import subprocess
def test_predict_cli():
    assert subprocess.call(["bash","-lc",". .venv/bin/activate && python src/predict.py --text 'great build quality'"]) == 0
