.PHONY: all data train evaluate app test clean

PYTHON = python

all: data train evaluate test

data:
	$(PYTHON) -m src.data

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

app:
	streamlit run app/streamlit_app.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
