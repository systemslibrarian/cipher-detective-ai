# Educational Boundary

Cipher Detective AI is a **teaching exhibit** for classical cryptanalysis. This
page is the canonical statement of what the project is — and what it is not.

## What this project is for

- Helping learners *see* how weak historical ciphers leak patterns.
- Showing transparent cryptanalytic signals (frequency, IoC, entropy, bigrams,
  Caesar/Atbash/Vigenère indicators) alongside an optional small classifier.
- Demonstrating a clean Hugging Face workflow: dataset → model → Space.
- Building intuition for *why* modern cryptography is structured the way it is.

## What this project is **not** for

- Breaking modern encryption (AES, ChaCha20, RSA, ECC, TLS, age, PGP, etc.).
- Recovering passwords or password hashes.
- Bypassing access controls, DRM, or authentication.
- Surveillance, deanonymization, or unauthorized access of any kind.
- Forensic tooling claims of any kind.
- Marketing claims like "military-grade", "AI hacking", "break any cipher".

If you came here looking for any of the above, this is the wrong project, and
that is intentional.

## Why this line matters

Real cryptography is not a vibe. It depends on:

- vetted primitives (peer-reviewed, standardized algorithms);
- correct **protocols** (key exchange, authentication, replay protection);
- careful **implementation** (constant-time, no padding oracles, safe RNG);
- key management (rotation, storage, recovery, revocation);
- metadata handling (sizes, timing, traffic analysis);
- an honest **threat model** for who, what, and when you are protecting against.

Classical ciphers fail every one of these by modern standards. That is exactly
what makes them excellent *teaching* material — and exactly why we keep the
framing honest.

## How we keep the line clear

- The Gradio app shows a persistent boundary banner.
- Dataset and model cards include "Not intended for" sections.
- Heuristic confidence is reported transparently and never inflated.
- Scope-creep PRs (e.g., "let's add a hash cracker") will be declined.
