## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

## Run
make train
make evaluate
make run-api
make app

## Tests
pytest -q -vv
