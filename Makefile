
PY := $(shell command -v python || command -v python3 || command -v py)
VENV=.venv
PIP=$(VENV)/bin/pip
PYBIN=$(VENV)/bin/python


export PYTHONPATH := .


setup:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt && $(PYBIN) -m nltk.downloader stopwords punkt
	touch src/__init__.py

train:
	$(PYBIN) -m src.train --data_path data/reviews_mapped.csv --model_dir models --test_size 0.2 --seed 42 --model logreg

evaluate:
	$(PYBIN) -m src.evaluate --model_dir models --out_dir outputs

inspect:
	$(PYBIN) -m src.inspect_artifacts

tune:
	$(PYBIN) -m src.tune --data_path data/reviews_mapped.csv --model_dir models

threshold:
	$(PYBIN) -m src.threshold --model_dir models --out models/threshold.txt

clean:
	rm -rf models/* outputs/*

.PHONY: run-app
run-app:
	$(PYBIN) -m streamlit run app/app.py

.PHONY: cv
cv:
	$(PYBIN) -m src.cv --data_path data/reviews_mapped.csv --out_dir outputs --model logreg --ngram_max 2 --class_weight balanced --stopwords_path data/domain_stopwords.txt

.PHONY: run-api
run-api:
	$(PYBIN) -m uvicorn api.main:app --host 0.0.0.0 --port 8000
