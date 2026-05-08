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

# 🕵️‍♂️ Cipher Detective AI

> **See the pattern. Test the hypothesis. Break the weak cipher. Respect the strong ones.**

Cipher Detective AI is an **educational** classical-cryptanalysis exhibit. It teaches how weak historical ciphers leak patterns, how cryptanalysis actually works, and **why modern cryptography is fundamentally different.**

It is built as a Hugging Face-native triple:

| Artifact | Suggested repo                                              | Role                                         |
|----------|-------------------------------------------------------------|----------------------------------------------|
| Space    | `systemslibrarian/cipher-detective-ai`                      | Live interactive exhibit (this app)          |
| Dataset  | `systemslibrarian/classical-cipher-corpus`                  | Labeled classical-cipher examples            |
| Model    | `systemslibrarian/cipher-detective-classifier`              | Small Transformer classifier                 |

**This is not an offensive tool.** It does not break modern encryption, recover passwords, or bypass access controls. See [`docs/educational-boundary.md`](docs/educational-boundary.md).

---

## ✨ Demo

| Detect Mode | Explain Mode | Compare Mode |
|---|---|---|
| ![Detect](screenshots/detect-mode.png) | ![Explain](screenshots/explain-mode.png) | ![Compare](screenshots/compare-mode.png) |

> Drop your screenshots into [`screenshots/`](screenshots/) — the README references `detect-mode.png`, `explain-mode.png`, `compare-mode.png`, and `challenge-mode.png`.

---

## 🧭 Modes

The Gradio Space ships with **five** modes:

1. **Detect Mode** — paste ciphertext, get a classification, confidence, and a full evidence report (frequency, IoC, entropy, Caesar/Affine candidates, Kasiski/Friedman indicators, transposition signal).
2. **Explain Mode** — see the raw "evidence notebook" without a verdict — useful for teaching.
3. **Challenge Mode** — generate practice ciphertexts (Caesar, Atbash, Vigenère, Rail-Fence, Columnar, Affine, Substitution) at chosen difficulty.
4. **Compare Mode** — run the **transparent heuristic baseline** and the **Transformer classifier** side-by-side, with disagreement analysis.
5. **About / Model Status** — live status of the loaded model, dataset/model repo references, and the educational-boundary statement.

---

## 🚀 Run locally

```bash
git clone https://github.com/systemslibrarian/cipher-detective-ai.git
cd cipher-detective-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The Space will be at <http://localhost:7860>.

To attach a trained Transformer classifier:

```bash
export CIPHER_MODEL_ID=systemslibrarian/cipher-detective-classifier   # or a local folder path
python app.py
```

If the model can't be loaded, the app **always** falls back to the transparent heuristic baseline.

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage spans cipher round-trips, edge cases (empty / non-alpha input), feature signals (IoC, entropy, Kasiski, Friedman, transposition), the heuristic classifier, and the dataset generator schema.

---

## 🗂️ Generate the dataset

Quick demo (5,000 rows):

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 5000 --seed 42
```

Public-release size (50,000 rows):

```bash
python scripts/generate_dataset.py --out data/cipher_examples.jsonl --n 50000 --seed 42
```

Each row includes `id`, `text`, `ciphertext`, `plaintext`, `label`, `cipher`, `key`, `difficulty`, `language`, `text_length`, `attack_methods`, and `educational_note` — see [`hf_cards/dataset_README.md`](hf_cards/dataset_README.md).

---

## 🧠 Train the model

```bash
python scripts/train_transformer.py \
  --data data/cipher_examples.jsonl \
  --model distilbert-base-uncased \
  --out cipher_model \
  --epochs 3
```

Outputs land in `cipher_model/`:

- model + tokenizer
- `training_metrics.json` (accuracy, macro precision/recall/F1)
- `label_mapping.json` (`label2id` / `id2label`)

To upload to the Hub afterwards:

```bash
huggingface-cli login
huggingface-cli upload systemslibrarian/cipher-detective-classifier ./cipher_model
```

---

## 📊 Evaluate

Heuristic baseline only:

```bash
python scripts/evaluate_baseline.py --data data/cipher_examples.jsonl --out reports/baseline_metrics.json
```

Compare heuristic vs Transformer:

```bash
python scripts/evaluate_baseline.py \
  --data data/cipher_examples.jsonl \
  --model cipher_model \
  --out reports/baseline_metrics.json
```

Report includes accuracy, macro F1, per-class precision/recall/F1, confusion matrix, and the dataset's label distribution.

---

## 🏷️ Labels

`plaintext`, `caesar_rot`, `atbash`, `vigenere`, `rail_fence`, `columnar`, `affine`, `substitution`.

---

## 🛣️ Roadmap

- [ ] Publish `classical-cipher-corpus` dataset (50k rows).
- [ ] Train and publish `cipher-detective-classifier`.
- [ ] Add `screenshots/` images.
- [ ] Add hill-climbing solver demo for monoalphabetic substitution (educational only).
- [ ] Per-length and per-difficulty evaluation buckets.
- [ ] Calibration plot (heuristic confidence vs accuracy).
- [ ] Multilingual plaintext sources (clearly labeled).
- [ ] Linked exhibit pages from Cipher Museum / Crypto Lab.

See [`CHANGELOG.md`](CHANGELOG.md) for what's already shipped.

---

## 🌐 Ecosystem

Cipher Detective AI is part of a broader cryptography-education path:

- **Cipher Museum** — curated history of ciphers _(link placeholder)_
- **Crypto Compare** — algorithm comparisons _(link placeholder)_
- **Crypto Lab** — hands-on experimentation _(link placeholder)_
- **Meow Decoder** — friendly entry point _(link placeholder)_

See [`docs/ecosystem.md`](docs/ecosystem.md).

---

## 🛡️ Educational boundary

This project teaches classical cryptanalysis. It is **not** designed to:

- break modern encryption (AES, ChaCha20, RSA, ECC, TLS, age, PGP),
- recover passwords or password hashes,
- bypass access controls or DRM,
- support surveillance or unauthorized access,
- make claims about real-world cryptographic security.

Modern cryptography depends on vetted primitives, protocols, key management, implementation correctness, metadata handling, and an honest threat model. None of the techniques shown here apply to it. See [`docs/educational-boundary.md`](docs/educational-boundary.md).

---

## 🤝 Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Honest, transparent, education-first contributions are very welcome.

## 🔐 Security

See [`SECURITY.md`](SECURITY.md) for how to report a vulnerability.

## 📜 License

MIT — see [`LICENSE`](LICENSE).

## 📚 Citation

If you use this in teaching or writing, see [`CITATION.cff`](CITATION.cff).

