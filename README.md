---
title: Cipher Detective AI
emoji: 🕵️‍♂️
colorFrom: indigo
colorTo: amber
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - cryptography
  - cryptanalysis
  - classical-ciphers
  - transformers
  - text-classification
  - cybersecurity-education
  - gradio
---

# Cipher Detective AI

**See the pattern. Test the hypothesis. Break the weak cipher. Respect the strong ones.**

Cipher Detective AI is a Hugging Face-native educational cryptanalysis project. It is designed as a public learning exhibit, not just a hosted web app.

It combines:

1. **Transparent cryptanalysis signals** — frequency analysis, index of coincidence, entropy, Caesar candidates, Atbash checks, n-gram evidence.
2. **A Transformer classification path** — trainable with `scripts/train_transformer.py` and deployable as a Hugging Face model.
3. **A dataset-first workflow** — reproducible generation of labeled classical-cipher examples.
4. **A museum-style learning experience** — Detect Mode, Explain Mode, and Challenge Mode.

## Why this belongs on Hugging Face

This project is meant to become three linked Hugging Face artifacts:

| Artifact | Suggested repo | Purpose |
|---|---|---|
| Space | `systemslibrarian/cipher-detective-ai` | Live interactive exhibit |
| Dataset | `systemslibrarian/classical-cipher-corpus` | Labeled classical cipher dataset |
| Model | `systemslibrarian/cipher-detective-classifier` | Transformer classifier trained on the corpus |

That makes it a real ML education project: dataset, model, evaluation, demo, and documentation.

## Modes

### Detect Mode

Paste ciphertext and receive:

- likely cipher family
- prediction score
- evidence report
- frequency table
- Caesar / ROT candidates
- natural-language reasoning
- clear limitations

### Explain Mode

Shows the evidence notebook without overclaiming:

- index of coincidence
- entropy
- top letters
- top bigrams/trigrams
- clues a human cryptanalyst would inspect

### Challenge Mode

Generates educational cipher challenges for learners.

Supported challenge families:

- Caesar / ROT
- Atbash
- Vigenère
- Rail Fence
- Columnar Transposition
- Affine

## Educational boundary

This project teaches classical cryptanalysis. It is **not** designed to:

- break modern encryption
- recover passwords
- bypass access controls
- assist surveillance
- support unauthorized access
- claim real-world cryptographic capability

Modern cryptography depends on vetted primitives, protocols, key management, implementation correctness, metadata handling, and threat modeling.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Generate a dataset

Quick demo dataset:

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 5000
```

Public-release dataset:

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 50000 --seed 42
```

## Evaluate the heuristic baseline

```bash
python scripts/evaluate_baseline.py --data data/cipher_examples.jsonl --out reports/baseline_metrics.json
```

## Train the Transformer classifier

```bash
python scripts/train_transformer.py \
  --data data/cipher_examples.jsonl \
  --model distilbert-base-uncased \
  --out cipher_model \
  --epochs 3
```

To use a hosted model in the Space, set this environment variable:

```text
CIPHER_MODEL_ID=systemslibrarian/cipher-detective-classifier
```

## Dataset labels

- `plaintext`
- `caesar_rot`
- `atbash`
- `vigenere`
- `rail_fence`
- `columnar`
- `affine`
- `substitution`

## Project posture

This project is part of a broader cryptography education path:

> Learn historical ciphers, compare algorithms, experiment with cryptographic ideas, and then apply secure engineering with honest threat models.

Cipher Detective AI focuses on the missing bridge: **how weak ciphers leak patterns and how those clues are discovered.**

## Quality checklist before a major public launch

- [ ] Generate 50k+ examples.
- [ ] Publish `classical-cipher-corpus` Dataset repo.
- [ ] Train `cipher-detective-classifier`.
- [ ] Publish model card with accuracy, macro F1, confusion matrix, and limitations.
- [ ] Set `CIPHER_MODEL_ID` in the Space.
- [ ] Add screenshots/GIFs to the README.
- [ ] Link from Cipher Museum, Crypto Lab, and GitHub profile.
- [ ] Add examples it gets wrong and explain why.

## License

MIT.
