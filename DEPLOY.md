# Deploy Guide

## 1. Create three Hugging Face repos

Recommended:

- Space: `systemslibrarian/cipher-detective-ai`
- Dataset: `systemslibrarian/classical-cipher-corpus`
- Model: `systemslibrarian/cipher-detective-classifier`

## 2. Push the Space

```bash
git init
git add .
git commit -m "Initial public-ready Cipher Detective AI Space"
git branch -M main
git remote add origin https://huggingface.co/spaces/systemslibrarian/cipher-detective-ai
git push -u origin main
```

## 3. Generate and publish the dataset

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 50000 --seed 42
```

Create a Dataset repo and upload:

- `data/cipher_examples.jsonl`
- `hf_cards/dataset_README.md` as `README.md`

## 4. Train and publish the model

```bash
python scripts/train_transformer.py \
  --data data/cipher_examples.jsonl \
  --model distilbert-base-uncased \
  --out cipher_model \
  --epochs 3
```

Upload `cipher_model/` to the model repo and use `hf_cards/model_README.md` as the model card.

## 5. Connect the Space to the model

In the Space settings, set:

```text
CIPHER_MODEL_ID=systemslibrarian/cipher-detective-classifier
```

The Space will then use the hosted Transformer model. If the model is unavailable, it falls back to transparent heuristics.
