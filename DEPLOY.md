# Deploy Guide

This guide walks through publishing the **Space**, the **dataset**, and the **model** so the project lives as three linked Hugging Face artifacts.

## 0. One-time setup

```bash
pip install -r requirements.txt
pip install -U huggingface_hub
huggingface-cli login
```

Pick a single Hub username (e.g. `systemslibrarian`) and use it for all three repos so they cross-link cleanly.

---

## 1. Create the three Hub repos

| Type     | Suggested ID                                       |
|----------|----------------------------------------------------|
| Space    | `systemslibrarian/cipher-detective-ai`             |
| Dataset  | `systemslibrarian/classical-cipher-corpus`         |
| Model    | `systemslibrarian/cipher-detective-classifier`     |

Create them in the Hub UI or via:

```bash
huggingface-cli repo create cipher-detective-ai --type space --space_sdk gradio
huggingface-cli repo create classical-cipher-corpus --type dataset
huggingface-cli repo create cipher-detective-classifier --type model
```

---

## 2. Push the Space

```bash
git init
git add .
git commit -m "Initial public-ready Cipher Detective AI Space"
git branch -M main
git remote add space https://huggingface.co/spaces/systemslibrarian/cipher-detective-ai
git push space main
```

(You can also keep GitHub as `origin` and use `space` as a second remote.)

---

## 3. Generate and publish the dataset

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 50000 --seed 42

# Inspect the schema:
head -n 1 data/cipher_examples.jsonl | python -m json.tool

# Upload to the dataset repo:
huggingface-cli upload systemslibrarian/classical-cipher-corpus \
    data/cipher_examples.jsonl \
    --repo-type dataset

# Publish the dataset card:
huggingface-cli upload systemslibrarian/classical-cipher-corpus \
    hf_cards/dataset_README.md README.md \
    --repo-type dataset
```

---

## 4. Train and publish the model

```bash
python scripts/train_transformer.py \
  --data data/cipher_examples.jsonl \
  --model distilbert-base-uncased \
  --out cipher_model \
  --epochs 3

# `cipher_model/` now contains the model, tokenizer,
# `training_metrics.json`, and `label_mapping.json`.

huggingface-cli upload systemslibrarian/cipher-detective-classifier ./cipher_model
huggingface-cli upload systemslibrarian/cipher-detective-classifier \
    hf_cards/model_README.md README.md
```

Optionally re-evaluate against the dataset for a published metrics snapshot:

```bash
python scripts/evaluate_baseline.py \
  --data data/cipher_examples.jsonl \
  --model cipher_model \
  --out reports/baseline_metrics.json
```

Commit `reports/baseline_metrics.json` (or attach it to the model card) so users can see the same numbers you do.

---

## 5. Connect the Space to the model

In the Space settings, add an environment variable:

```text
CIPHER_MODEL_ID=systemslibrarian/cipher-detective-classifier
```

Restart the Space. The "About / Model Status" tab will confirm the model loaded.
If the model is unavailable for any reason, the Space falls back to the
transparent heuristic baseline — by design.

---

## 6. Pre-launch checklist

- [ ] Screenshots in [`screenshots/`](screenshots/) and referenced in [`README.md`](README.md).
- [ ] `data/cipher_examples.jsonl` published with a dataset card.
- [ ] `cipher_model/` published with a model card and metrics.
- [ ] `CIPHER_MODEL_ID` set in the Space.
- [ ] Cross-links between Space ↔ Dataset ↔ Model.
- [ ] Educational-boundary banner visible in the Space.
- [ ] `pytest -q` passes locally.

