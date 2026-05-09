---
pretty_name: Classical Cipher Corpus
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - cryptography
  - classical-ciphers
  - cryptanalysis
  - cybersecurity-education
  - synthetic-data
size_categories:
  - 10K<n<100K
---

# Classical Cipher Corpus

A labeled educational dataset of classical cipher examples for teaching cryptanalysis and training small cipher-family classifiers.

**Part of the Cipher Detective AI project:**
- 🕵️ Space: [systemslibrarian/cipher-detective-ai](https://huggingface.co/spaces/systemslibrarian/cipher-detective-ai)
- 📦 Dataset: [systemslibrarian/classical-cipher-corpus](https://huggingface.co/datasets/systemslibrarian/classical-cipher-corpus) _(this repo)_
- 🤖 Model: [systemslibrarian/cipher-detective-classifier](https://huggingface.co/systemslibrarian/cipher-detective-classifier)

## Intended use

- Teach classical cryptanalysis.
- Benchmark educational cipher-family detectors.
- Train small text classifiers for Hugging Face Spaces.
- Demonstrate why historical ciphers are not modern security.

## Not intended for

- Unauthorized access.
- Surveillance.
- Breaking modern encryption.
- Password recovery.
- Bypassing security controls.
- Real-world cryptographic security claims.

## Schema

Each line of `cipher_examples.jsonl` is one record:

```json
{
  "id": "cda-0000042",
  "text": "WKLV LV D FODVVLFDO FDHVDU FLSKHU GHPR",
  "ciphertext": "WKLV LV D FODVVLFDO FDHVDU FLSKHU GHPR",
  "plaintext": "THIS IS A CLASSICAL CAESAR CIPHER DEMO",
  "label": "caesar_rot",
  "cipher": "caesar_rot",
  "key": {"shift": 3},
  "difficulty": "medium",
  "language": "en",
  "text_length": 38,
  "length": 38,
  "attack_methods": ["brute_force_26", "frequency_analysis", "chi_squared_english"],
  "educational_note": "Caesar / ROT-N is a single-shift monoalphabetic cipher with only 26 keys.",
  "source": "synthetic_educational"
}
```

`text` and `ciphertext` are kept as aliases for compatibility. `text_length` and `length` likewise.

## Labels

- `plaintext`
- `caesar_rot`
- `atbash`
- `vigenere`
- `rail_fence`
- `columnar`
- `affine`
- `substitution`

## Construction

The dataset is generated synthetically from educational English plaintext templates and classical cipher transforms. It is reproducible with:

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 50000 --seed 42
```

## Biases and limitations

- Mostly English.
- Synthetic text.
- Limited cipher families.
- Not representative of modern encryption.
- Not a benchmark for real-world cryptanalytic capability.
- Model performance on this dataset should be framed only as educational classification performance.

## Responsible framing

This dataset exists to teach pattern leakage in classical ciphers. It should be used to make learners more skeptical of weak encryption claims and more respectful of modern, well-reviewed cryptography.
