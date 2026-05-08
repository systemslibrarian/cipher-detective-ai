---
license: mit
library_name: transformers
tags:
  - text-classification
  - cryptography
  - cryptanalysis
  - classical-ciphers
  - cybersecurity-education
datasets:
  - systemslibrarian/classical-cipher-corpus
metrics:
  - accuracy
  - f1
---

# Cipher Detective Classifier

A Transformer-based educational classifier for identifying broad classical-cipher families from ciphertext-like text.

## Intended use

This model supports the **Cipher Detective AI** Hugging Face Space. It is intended for:

- cryptography education
- classical cipher demonstrations
- comparison against transparent heuristic baselines
- teaching why weak ciphers leak patterns

## Labels

- `plaintext`
- `caesar_rot`
- `atbash`
- `vigenere`
- `rail_fence`
- `columnar`
- `affine`
- `substitution`

## Not intended for

- breaking modern encryption
- unauthorized access
- surveillance
- password recovery
- bypassing security controls
- claims about real-world cryptanalytic capability

## Training

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 50000 --seed 42
python scripts/train_transformer.py \
  --data data/cipher_examples.jsonl \
  --model distilbert-base-uncased \
  --out cipher_model \
  --epochs 3
```

## Evaluation

After training, include:

- accuracy
- macro F1
- per-label precision/recall/F1
- confusion matrix
- examples of false positives
- examples of false negatives

## Limitations

The model learns patterns from synthetic educational examples. It can confuse cipher families with similar surface statistics, especially transposition, substitution, and short Vigenère samples. It should be presented as a teaching classifier, not as a cryptanalytic authority.

## Responsible-use statement

Use this model to teach how classical ciphers leak patterns and why modern cryptography requires careful protocol design, key management, implementation correctness, and threat modeling.
