setup:
	pip install -e .
train:
	python -m dev.train --epochs 3
test:
	pytest -q
serve:
	uvicorn api.main:app --reload
