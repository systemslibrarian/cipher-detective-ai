# Security Policy

## Scope

Cipher Detective AI is an **educational** project for teaching classical cryptanalysis. It is intentionally not a security product. It does not:

- protect data,
- store secrets,
- handle authentication,
- break modern encryption,
- recover passwords,
- bypass access controls.

If you are looking for production cryptography, use vetted, well-reviewed libraries and protocols (e.g., libsodium, Noise, TLS 1.3, age, modern KMS systems).

## Reporting a vulnerability

If you find a security issue in this repository (e.g., supply-chain risk, dependency vulnerability, code execution path in the Gradio app, unsafe file handling), please report it privately:

1. Open a **private security advisory** on GitHub, or
2. Email the maintainer listed on the GitHub profile.

Please do **not** open a public issue for security reports.

## What is in scope

- Code execution / unsafe deserialization in `app.py`, `core.py`, or scripts.
- Path-traversal or file-write issues in dataset/training scripts.
- Dependency vulnerabilities affecting users who install `requirements.txt`.
- Secrets accidentally committed to the repository.

## What is out of scope

- The fact that classical ciphers are weak (this is the entire point of the project).
- Performance limitations of the heuristic baseline.
- The model misclassifying samples — please file these as normal issues.

## Responsible disclosure

We aim to acknowledge reports within 7 days and to publish a fix or mitigation within 30 days for confirmed issues, faster when severity warrants.
