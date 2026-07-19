# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Cipher-family layer**: `CIPHER_FAMILIES` mapping and `label_family()` in
  `core.py` group all labels into seven statistical families (`plain`,
  `mono_substitution`, `polyalphabetic`, `transposition`, `polygraphic`,
  `machine_or_otp`, `code_format`). Family-level accuracy/macro-F1/confusion
  matrix now reported by `evaluate_baseline.py` as the honest headline metric —
  many fine labels (all rotor machines, kama_sutra vs monoalphabetic) are
  mathematically indistinguishable from ciphertext alone. The Detect
  explanation shows the family verdict alongside the fine label.
- `scripts/inspect_metrics.py`: summarize an evaluation report (worst classes,
  top confusions, family metrics, length buckets).
- Tests: Beaufort known-vector + reciprocity (previously zero coverage),
  rail-fence and columnar round-trips, Porta reciprocity, lorenz vs
  null_cipher disambiguation, family-mapping completeness.
- CI: auto-sync to Hugging Face Space on push to main
- **Solve Substitution tab**: hill-climbing solver for monoalphabetic substitution,
  seeded from observed letter-frequency rank, scored by English bigram log-prob,
  with random restarts to escape local optima. Educational only.
- **"Random example" button** in Detect Mode for one-click demos.
- `evaluate_baseline.py`: per-difficulty and per-length-bucket accuracy + macro-F1
  breakdowns alongside the overall metrics.
- `requirements.txt`: explicit `torch>=2.2.0` pin so the Transformer pipeline
  actually loads on a fresh Hugging Face Space.
- `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, `CHANGELOG.md`.
- `requirements-dev.txt`, `pyproject.toml`, `Makefile` for reproducible dev workflows.
- `docs/` folder: README, cryptanalysis cheatsheet, educational-boundary, ecosystem.
- `screenshots/` placeholder folder.
- `core.py`: Kasiski examination, Friedman key-length estimate, transposition signal,
  brute-force Affine candidates, `vigenere_decrypt`, `substitution_encrypt`.
- `app.py`: **Compare Mode** — heuristic vs Transformer side-by-side with disagreement
  highlighting; richer Explain Mode with Friedman / Kasiski / transposition signals;
  Substitution added to Challenge Mode.
- Dataset generator: rich Hugging Face-friendly schema (`id`, `cipher`, `key`,
  `difficulty`, `language`, `text_length`, `attack_methods`, `educational_note`).
- `evaluate_baseline.py`: optional `--model` flag to compare a Transformer
  classifier against the heuristic baseline; richer JSON report including
  per-class metrics, confusion matrix, and label distribution.
- Tests: cipher round-trips, edge cases (empty / non-alpha / short input),
  feature-signal sanity (IoC, entropy, Kasiski, Friedman, transposition),
  heuristic label correctness for Caesar / Atbash / plaintext, and dataset
  schema + reproducibility (`--seed`) checks.

### Fixed
- **`columnar_transposition_decrypt` returned scrambled text**: ciphertext
  segments were read back in original-column order instead of key-sorted order
  (the order encrypt writes them), and irregular column lengths were computed
  from sort rank instead of grid position. Caught by the new round-trip tests;
  the keyed-columnar option in Try Decode produced wrong output before this.
- **Porta generator was not a valid cipher**: the hand-typed 13-row table
  contained a non-permutation row (two `A`s, no `M`) and didn't implement real
  Porta. Replaced with the authentic self-reciprocal formula — verified to
  match the museum corpus' porta rows exactly.
- **Windows console crash (`UnicodeEncodeError`)** in `generate_dataset.py`,
  `balance_dataset.py`, and `convert_museum_corpus.py`: `→` in output can't be
  encoded by the default cp1252 console; stdout is now reconfigured to UTF-8.
  This made two tests fail on Windows.
- **Heuristic misrouting** (all confirmed against the test-split confusion
  matrix, +2 points fine-grained accuracy on a fixed 10k sample):
  - `null_cipher` rule now requires spaces, so run-together near-English is
    classified `lorenz` (94 rows previously leaked).
  - Removed the `kama_sutra` rule — it stole 5× more monoalphabetic/caesar/
    affine rows than it correctly claimed (the two ciphers are statistically
    identical; kama_sutra now resolves at family level).
  - Removed the `IoC ≥ 0.063 → stager_route` gate (right 21×, wrong ~600×;
    IoC differences between transpositions are sampling noise).
  - Exotic machine-class verdicts (slidex, jefferson_disk, purple, diana,
    red_type_a, fialka…) now require ≥ 60 letters — below that, chi/IoC are
    too noisy and short polyalphabetic texts were being misrouted.
  - `wheatstone` rule now requires run-together text, so spaced substitution
    ciphertexts fall to `monoalphabetic`.
- `evaluate_baseline.py`: family-level metrics now score raw predictions, so
  `too_short` counts as an honest miss (family `unknown`) instead of being
  remapped to `plaintext` — the remap both polluted the plain family with
  false positives and silently dropped those rows from the confusion matrix
  (whose label set lacks `plaintext`), inflating matrix-derived metrics.
- **Boot-blocker**: removed unsupported `aria_label=` kwargs from `gr.Textbox`,
  `gr.Dropdown`, and `gr.Button` calls — these crash Gradio 4.44 on launch.
  Accessibility names are still provided through `label=` and `info=`.

### Changed
- `README.md` rewritten as a public-launch README: hero, modes, screenshots,
  HF Space instructions, dataset/model workflow, evaluation, roadmap,
  ecosystem, educational boundary, contributing, security, license, citation.
- `DEPLOY.md` expanded into a step-by-step Hugging Face publish guide.
- `hf_cards/dataset_README.md` updated to the new dataset schema.
- `examples/sample_ciphertexts.md` expanded (8 samples + answer key).
- `requirements.txt` adds `huggingface_hub`; runtime/dev split documented.
- Heuristic classifier now reports `rail_fence` and `columnar` separately
  instead of a single generic `transposition` label.
- `.gitignore` expanded for common Python / Hugging Face artifacts.

## [0.1.0] — 2026-05-08

Initial public foundation.
