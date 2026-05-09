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

**Part of the Cipher Detective AI project:**
- 🕵️ Space: [systemslibrarian/cipher-detective-ai](https://huggingface.co/spaces/systemslibrarian/cipher-detective-ai)
- 📦 Dataset: [systemslibrarian/classical-cipher-corpus](https://huggingface.co/datasets/systemslibrarian/classical-cipher-corpus)
- 🤖 Model: [systemslibrarian/cipher-detective-classifier](https://huggingface.co/systemslibrarian/cipher-detective-classifier) _(this repo)_

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
python scripts/convert_museum_corpus.py \
  --museum /path/to/cipher-museum/public/corpus/all.jsonl \
  --out data/cipher_examples.jsonl --split all
python scripts/train_transformer.py \
  --data data/cipher_examples.jsonl \
  --model distilbert-base-uncased \
  --out cipher_model \
  --epochs 3 --batch-size 64
```

Training data: **100,026 examples** from [systemslibrarian/classical-cipher-corpus](https://huggingface.co/datasets/systemslibrarian/classical-cipher-corpus) — sourced from the [Cipher Museum](https://ciphermuseum.com) public corpus (CC0).

## Evaluation (3 epochs, 20% held-out test set)

| Metric | Value |
|---|---|
| Accuracy | **48.3%** |
| Macro Precision | **56.9%** |
| Macro Recall | **52.0%** |
| Macro F1 | **51.8%** |

81-class classification (random baseline ≈ 1.2%). Performance varies significantly by cipher family — substitution families with distinctive patterns (Caesar, Atbash, Morse) score highest; machine ciphers with similar statistical profiles (Enigma, Lorenz, SIGABA) are harder to distinguish from surface text alone.

## Limitations

The model learns surface statistical patterns from the ciphertext text field. It can confuse cipher families with similar statistics, especially machine ciphers, polyalphabetic variants, and short samples. It should be presented as a teaching classifier, not as a cryptanalytic authority.

## Responsible-use statement

Use this model to teach how classical ciphers leak patterns and why modern cryptography requires careful protocol design, key management, implementation correctness, and threat modeling.
