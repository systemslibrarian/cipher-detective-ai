---
title: Cipher Detective AI
emoji: 🕵️‍♂️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - cryptography
  - cryptanalysis
  - classical-ciphers
  - cipher
  - transformers
  - text-classification
  - cybersecurity-education
  - gradio
  - nlp
  - machine-learning
  - python
  - substitution-cipher
  - vigenere
  - caesar-cipher
  - educational
---

# 🕵️‍♂️ Cipher Detective AI

> **See the pattern. Test the hypothesis. Break the weak cipher. Respect the strong ones.**

Cipher Detective AI is an **educational** classical-cryptanalysis exhibit. It detects and decodes **81 historical cipher types** — from Caesar and Vigenère to Rail-Fence, Columnar, Playfair, and more — using a transparent heuristic engine and an optional fine-tuned Transformer classifier. It teaches how weak historical ciphers leak patterns, how cryptanalysis actually works, and **why modern cryptography is fundamentally different.**

It is built as a Hugging Face-native triple:

| Artifact | Repo                                              | Role                                         |
|----------|-------------------------------------------------------------|----------------------------------------------|
| Space    | [systemslibrarian/cipher-detective-ai](https://huggingface.co/spaces/systemslibrarian/cipher-detective-ai) | Live interactive exhibit (this app) |
| Dataset  | [systemslibrarian/classical-cipher-corpus](https://huggingface.co/datasets/systemslibrarian/classical-cipher-corpus) | Labeled classical-cipher examples |
| Model    | [systemslibrarian/cipher-detective-classifier](https://huggingface.co/systemslibrarian/cipher-detective-classifier) | Small Transformer classifier |

**This is not an offensive tool.** It does not break modern encryption, recover passwords, or bypass access controls. See [`docs/educational-boundary.md`](docs/educational-boundary.md).

---

## ✨ Demo

> Open the live Space to try it: <https://huggingface.co/spaces/systemslibrarian/cipher-detective-ai>
> Static screenshots live in [`screenshots/`](screenshots/) once captured (the
> README intentionally avoids broken `<img>` tags before they exist).

---

## 🧭 Modes

The Gradio Space ships with **seven** tabs:

1. **Detect** — paste ciphertext, get a classification, confidence, and a full evidence report (frequency, IoC, entropy, Caesar/Affine candidates, Kasiski/Friedman indicators, transposition signal). One-click random examples.
2. **Evidence Notebook** — see the raw evidence without a verdict — useful for teaching step-by-step cryptanalysis.
3. **Challenge** — generate practice ciphertexts (Caesar, Atbash, Vigenère, Rail-Fence, Columnar, Affine, Substitution) at chosen difficulty.
4. **Try Decode** — twelve decryption methods with automatic English-quality scoring:
   - *Auto modes (no key needed):* auto-best-Caesar, auto-best-Affine, auto-Vigenère (Kasiski + Friedman + key refinement + keyword dictionary), auto-Beaufort, auto-columnar (column-order hill climb), auto-rail-fence (brute-force rails 2–15)
   - *Keyed modes:* Caesar/ROT, Atbash, Vigenère, Beaufort, Affine, Columnar transposition
5. **Compare Mode** — run the **transparent heuristic baseline** and the **Transformer classifier** side-by-side, with disagreement analysis.
6. **Solve Substitution** — greedy n-gram descent solver for monoalphabetic substitution, scored by a blended bigram/trigram/quadgram log-probability model. Educational only — solves ~97% of 300-letter English samples, degrades on shorter text.
7. **About** — project background, educational boundaries, and links.

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

Report includes accuracy, macro F1, per-class precision/recall/F1, confusion
matrix, family-level metrics, and the dataset's label distribution. Summarize
any report with:

```bash
python scripts/inspect_metrics.py reports/baseline_metrics.json
```

### Current heuristic baseline (30,104-row held-out test split, 81 classes)

| Metric | Fine-grained (81 classes) | Family-level (7 families) |
|---|---:|---:|
| Accuracy | 29.2% | 57.2% |
| Macro F1 | 0.393 | 0.656 |

Family-level is the honest headline: a large share of fine-grained error is
between classes that are mathematically indistinguishable from ciphertext
alone (see **Labels & cipher families** below). Accuracy also depends heavily
on length — near-perfect above 200 letters, weak under 50, which is itself an
accurate lesson about classical cryptanalysis.

### Auto-solver success rates (20 trials per cell, plaintext fully recovered)

| Solver | 40 letters | 80 | 160 | 300 |
|---|---:|---:|---:|---:|
| Caesar (brute force + chi²) | 100% | 100% | 100% | 100% |
| Affine (312-key brute force) | 45% | 100% | 100% | 100% |
| Vigenère (Kasiski/Friedman + key refinement) | 5% | 60% | 100% | 100% |
| Beaufort (reciprocal Vigenère) | 5% | 65% | 100% | 100% |
| Columnar (column-order hill climb) | 55% | 70% | 75% | 85% |
| Rail fence (rail brute force) | 50% | 100% | 100% | 100% |
| Substitution (greedy n-gram descent) | — | — | 60% | 95% |

Reproduce with `python scripts/benchmark_solvers.py`. Vigenère and Beaufort
are measured with keys **not** in the built-in keyword dictionary, so these
are the statistical attack's true rates — a common keyword cracks short texts
the dictionary pass covers separately. The short-text fall-off is the physics
of the problem, not a bug: 40 letters split across a Vigenère key leaves too
few letters per column for statistics to grip — a lesson the exhibit is happy
to teach. A `--assert-gate` mode runs these as a CI regression guard.

### Confidence calibration

The heuristic's raw confidences were badly miscalibrated: a raw 26% meant ~4%
real accuracy and a raw 86% meant ~97%. An isotonic map (fit on the validation
split by `scripts/calibrate_confidence.py`, applied inside `heuristic_classify`)
corrects this, cutting expected calibration error from **0.18 to 0.04** on the
held-out test split (map fit on validation, measured on test — a genuine
out-of-sample number), so a reported N% means the classifier is right about N%
of the time. Regenerate with:

```bash
python scripts/calibrate_confidence.py --data data/splits/val.jsonl
```

---

## 🏷️ Labels & cipher families

Fine-grained labels are grouped into **seven statistical families** (`plain`,
`mono_substitution`, `polyalphabetic`, `transposition`, `polygraphic`,
`machine_or_otp`, `code_format`). This matters because many fine labels are
**mathematically indistinguishable from ciphertext alone** — every well-built
rotor machine (Enigma, Typex, SIGABA, KL-7, Fialka…) emits a near-uniform
letter stream, and a Kama-Sutra cipher *is* a monoalphabetic substitution.
No classifier can honestly separate those; one that appears to is memorizing
generator artifacts. The evaluation therefore reports **family-level accuracy
as the headline metric** and fine-grained accuracy as best-effort within a
family (`label_family()` in `core.py` exposes the mapping). That limitation is
itself the lesson: indistinguishability from random *is the design goal of
good cryptography*.

The classifier covers **81 cipher classes**, including:

`plaintext`, `caesar_rot`, `atbash`, `vigenere`, `beaufort`, `rail_fence`, `columnar`, `affine`, `substitution`, `playfair`, `four_square`, `two_square`, `hill_cipher`, `bifid`, `trifid`, `adfgx`, `adfgvx`, `enigma`, `lorenz`, `morse_code`, `tap_code`, `navajo_code`, `pigpen`, `baconian`, `polybius`, `straddling_checkerboard`, `chaocipher`, `nihilist`, `porta`, `rot13`, `rot47`, `gronsfeld`, `running_key`, `autokey`, `one_time_pad`, `venona_pad_reuse`, `voynich`, `babington`, and more.

See [`data/cipher_examples.jsonl`](data/cipher_examples.jsonl) for the full label distribution.

---

## 🛣️ Roadmap

- [ ] Publish `classical-cipher-corpus` dataset — one command: `python scripts/upload_to_hub.py dataset --repo …`.
- [ ] Train and publish `cipher-detective-classifier` — needs a GPU Space; use `--char-level --family-labels`.
- [ ] Add `screenshots/` images (needs a running Gradio runtime to capture).
- [x] Confidence calibration: isotonic map so reported confidence matches real accuracy.
- [x] Beaufort and columnar auto-solvers + Vigenère keyword-dictionary pass.
- [x] Property-based round-trip tests for all cipher encoders.
- [x] CI: Windows runner + solver-regression gate.
- [x] Cipher-family layer: family-level accuracy reporting + `label_family()` mapping.
- [x] Hill-climbing solver demo for monoalphabetic substitution (educational only).
- [x] Per-length and per-difficulty evaluation buckets.
- [x] Vigenère auto-solver (Kasiski + Friedman key-length estimation).
- [x] Rail-fence and columnar transposition decoders.
- [x] Beaufort cipher support (encrypt + decrypt).
- [x] Bigram + trigram blended scoring for hill climber.
- [x] GitHub Actions → Hugging Face Space auto-sync on every push.
- [x] Confidence calibration (heuristic confidence vs measured accuracy).
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

