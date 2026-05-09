.PHONY: help install dev test lint format dataset dataset-large baseline train train-best app clean

help:
	@echo "Cipher Detective AI — common tasks"
	@echo ""
	@echo "  make install        Install runtime dependencies"
	@echo "  make dev            Install runtime + dev dependencies"
	@echo "  make test           Run the test suite"
	@echo "  make lint           Run ruff (lint only)"
	@echo "  make format         Run black + ruff --fix"
	@echo "  make dataset        Generate 10,000-row synthetic dataset (20 cipher types)"
	@echo "  make dataset-large  Augment existing dataset with 500 examples per cipher"
	@echo "  make baseline       Evaluate the heuristic baseline on the dataset"
	@echo "  make train          Quick fine-tune (distilroberta, 5 epochs)"
	@echo "  make train-best     Best-accuracy fine-tune (roberta-base, 10 epochs, focal loss)"
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
	python scripts/generate_dataset.py \
	  --out data/cipher_examples.jsonl \
	  --n 10000 \
	  --seed 42

dataset-large:
	python scripts/generate_dataset.py \
	  --out data/cipher_examples.jsonl \
	  --append \
	  --per-label 500 \
	  --seed 777 \
	  --labels beaufort gronsfeld autokey trithemius porta scytale \
	           double_transposition stager_route rot13 monoalphabetic

baseline:
	python scripts/evaluate_baseline.py \
	  --data data/cipher_examples.jsonl \
	  --out reports/baseline_metrics.json

train:
	python scripts/train_transformer.py \
	  --data data/cipher_examples.jsonl \
	  --model distilroberta-base \
	  --out cipher_model \
	  --epochs 5 \
	  --lr 3e-5

train-best:
	python scripts/train_transformer.py \
	  --data data/cipher_examples.jsonl \
	  --model roberta-base \
	  --out cipher_model \
	  --epochs 10 \
	  --batch-size 16 \
	  --grad-accum 2 \
	  --lr 2e-5 \
	  --warmup-ratio 0.06 \
	  --label-smoothing 0.05 \
	  --focal-loss

app:
	python app.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

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
