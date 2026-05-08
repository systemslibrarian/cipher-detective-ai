# Contributing to Cipher Detective AI

Thanks for your interest! This is an **educational** cryptanalysis project. Contributions that strengthen its teaching value, transparency, and correctness are very welcome.

## Ground rules

- Stay within the **classical-cipher / education** scope. We do not accept changes that frame the project as a tool to break modern encryption, recover passwords, or bypass access controls.
- Be honest about limitations. Heuristics and small classifiers are explicitly imperfect; documentation should reflect that.
- Keep the heuristic path **transparent and explainable**. Every signal should be inspectable by a learner.

## Development setup

```bash
git clone https://github.com/systemslibrarian/cipher-detective-ai.git
cd cipher-detective-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
```

## What we welcome

- New classical ciphers (with encoder + tests + dataset entry).
- Better heuristic features (Kasiski, Friedman, n-gram log-likelihoods, etc.).
- Improved Gradio UX and educational copy.
- Dataset card, model card, and documentation improvements.
- More principled evaluation (per-length buckets, calibration, error analysis).

## What we will likely decline

- Anything claiming real-world cryptanalytic capability against modern systems.
- Removing the safety / scope language from the UI or docs.
- Adding heavy dependencies that hurt Hugging Face Space cold-start time.

## Pull request checklist

- [ ] `pytest -q` passes locally.
- [ ] New behavior is covered by tests.
- [ ] Public-facing strings are honest and educational.
- [ ] README / docs updated when behavior changes.
- [ ] No copyrighted training text added.

## Code style

- Python 3.10+, type hints encouraged.
- Keep `core.py` import-light (no `gradio`, no `transformers`).
- Prefer small, named helpers over large lambdas.

## Reporting issues

When filing a bug, include:

- the input ciphertext (or a synthetic equivalent),
- the expected vs actual classification,
- the relevant evidence values from Explain Mode.
