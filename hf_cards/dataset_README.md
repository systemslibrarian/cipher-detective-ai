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

```json
{
  "text": "ciphertext or plaintext sample",
  "label": "caesar_rot",
  "plaintext": "known educational source text",
  "metadata": {"shift": 3},
  "length": 64,
  "source": "synthetic_educational"
}
```

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
