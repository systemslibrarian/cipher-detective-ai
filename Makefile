.PHONY: help install dev test lint format dataset dataset-large baseline train app clean

help:
	@echo "Cipher Detective AI — common tasks"
	@echo ""
	@echo "  make install        Install runtime dependencies"
	@echo "  make dev            Install runtime + dev dependencies"
	@echo "  make test           Run the test suite"
	@echo "  make lint           Run ruff (lint only)"
	@echo "  make format         Run black + ruff --fix"
	@echo "  make dataset        Generate a 5,000-row demo dataset"
	@echo "  make dataset-large  Generate a 50,000-row release dataset"
	@echo "  make baseline       Evaluate the heuristic baseline on the dataset"
	@echo "  make train          Fine-tune distilbert on the dataset"
	@echo "  make app            Launch the Gradio Space locally"
	@echo "  make clean          Remove caches and build artifacts"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .

format:
	black .
	ruff check --fix .

dataset:
	python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 5000 --seed 42

dataset-large:
	python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 50000 --seed 42

baseline:
	python scripts/evaluate_baseline.py --data data/cipher_examples.jsonl --out reports/baseline_metrics.json

train:
	python scripts/train_transformer.py \
	  --data data/cipher_examples.jsonl \
	  --model distilbert-base-uncased \
	  --out cipher_model \
	  --epochs 3

app:
	python app.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
