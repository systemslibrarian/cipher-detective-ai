# Deploy Guide

This guide walks through publishing the **Space**, the **dataset**, and the **model** so the project lives as three linked Hugging Face artifacts.

## 0. One-time setup

```bash
pip install -r requirements.txt
hf auth login
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
hf repos create cipher-detective-ai --type space --space-sdk gradio
hf repos create classical-cipher-corpus --type dataset
hf repos create cipher-detective-classifier --type model
```

---

## 2. Push the Space

```bash
git lfs install
git lfs track "*.jsonl"
git add .gitattributes
git commit -m "Track large files with git LFS"
git remote add space https://huggingface.co/spaces/systemslibrarian/cipher-detective-ai
git push space main
```

Note: HF rejects files > 10 MB unless tracked with git LFS. The `*.jsonl` pattern covers the dataset file. Keep GitHub as `origin` and use `space` as a second remote.

---

## 3. Generate and publish the dataset

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 50000 --seed 42

# Inspect the schema:
head -n 1 data/cipher_examples.jsonl | python -m json.tool

# Upload to the dataset repo:
hf upload systemslibrarian/classical-cipher-corpus \
    data/cipher_examples.jsonl --repo-type dataset

# Publish the dataset card:
hf upload systemslibrarian/classical-cipher-corpus \
    hf_cards/dataset_README.md README.md --repo-type dataset
```

---

## 4. Train and publish the model

### Easiest: free Colab GPU (no billing)

Open [`notebooks/train_on_colab.ipynb`](notebooks/train_on_colab.ipynb) in
[Google Colab](https://colab.research.google.com/), switch the runtime to the
free **T4 GPU**, paste a free Hugging Face **Write** token when asked, and run
the cells top to bottom. It trains the character-level classifier and pushes it
to the Hub — no paid GPU Space required. The notebook also uploads the model,
so you can skip straight to step 5.

### Or locally / on your own GPU

```bash
python scripts/train_transformer.py \
  --data data/splits/train.jsonl \
  --test-data data/splits/val.jsonl \
  --model distilbert-base-uncased \
  --char-level \
  --out cipher_model \
  --epochs 5

# `cipher_model/` now contains the model, tokenizer,
# `training_metrics.json`, and `label_mapping.json`.

hf upload systemslibrarian/cipher-detective-classifier ./cipher_model
hf upload systemslibrarian/cipher-detective-classifier \
    hf_cards/model_README.md README.md
```

`--char-level` tokenizes one character at a time — the right fit for ciphertext.
The flag is saved in the model config, and the app applies the same spacing at
inference automatically, so the model behaves identically in production and
training. On CPU a full run is slow; use Colab or a GPU box.

### Improving accuracy (higher-success recipes)

The baseline DistilBERT run reaches ~57% fine-grained accuracy (81 classes) and
tends to over-predict shift ciphers (Caesar/Vigenère/Trithemius look alike at
the character level). Two drop-in upgrades, both run on the same free Colab GPU:

**Stronger model + focal loss** — bigger backbone and a loss that fights the
class imbalance driving the shift-cipher bias:

```bash
python scripts/train_transformer.py \
  --data data/splits/train.jsonl --test-data data/splits/val.jsonl \
  --model roberta-base --char-level --focal-loss \
  --epochs 10 --batch-size 32 --out cipher_model \
  --push-to-hub --hub-model-id <you>/cipher-detective-classifier
```

**Character-native model (CANINE)** — operates directly on Unicode codepoints,
so it sees position/periodicity without the spacing trick (drop `--char-level`;
CANINE is char-native and the app feeds it raw text automatically):

```bash
python scripts/train_transformer.py \
  --data data/splits/train.jsonl --test-data data/splits/val.jsonl \
  --model google/canine-s --focal-loss \
  --epochs 10 --max-length 512 --out cipher_model \
  --push-to-hub --hub-model-id <you>/cipher-detective-classifier
```

After publishing a new model, run `scripts/calibrate_transformer.py` and commit
the refreshed `transformer_calibration_map.json` so its reported confidence
still matches its accuracy.

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

