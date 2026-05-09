# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
