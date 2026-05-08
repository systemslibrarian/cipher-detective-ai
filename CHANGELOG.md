# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LICENSE` (MIT).
- `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, `CHANGELOG.md`.
- `requirements-dev.txt`, `pyproject.toml`, `Makefile` for reproducible dev workflows.
- `docs/` folder with extended educational notes.
- `screenshots/` placeholder folder.

### Changed
- README expanded with screenshots block, ecosystem links, dataset/model
  workflow, and explicit educational boundary.
- Heuristic classifier adds Kasiski / Friedman style indicators for Vigenère.
- Dataset generator emits a richer, Hugging Face-friendly schema
  (`id`, `cipher`, `key`, `difficulty`, `language`, `text_length`,
  `attack_methods`, `educational_note`).
- Trainer logs macro precision / recall / F1 and persists `label_mapping.json`.

### Notes
- This release is the first public-launch candidate. No public versions
  existed prior to this point; everything pre-`0.1.0` should be considered
  internal scaffolding.

## [0.1.0] — 2026-05-08

Initial public foundation.
