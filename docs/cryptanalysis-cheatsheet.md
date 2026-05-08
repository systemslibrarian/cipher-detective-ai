# Classical Cryptanalysis Cheat Sheet

These are the signals Cipher Detective AI surfaces in **Explain Mode** and uses
inside the heuristic baseline. None of them are proofs; they are clues.

## Frequency analysis

English text has a very uneven letter distribution. `E T A O I N S H R` cover
roughly 70% of letters in normal prose. A monoalphabetic substitution preserves
that *shape* but relabels the letters — so the histogram still looks "spiky",
just with different letters on top.

## Index of coincidence (IoC)

$$
\mathrm{IoC} = \frac{\sum_i n_i (n_i - 1)}{N (N-1)}
$$

Rough reference values:

| Text type                          | Typical IoC |
|------------------------------------|-------------|
| English plaintext                  | ~0.066      |
| Monoalphabetic substitution        | ~0.066      |
| Random uniform letters             | ~0.038      |
| Vigenère with long key             | ~0.040–0.045 |

If the IoC is close to English, the cipher is likely **plaintext, Caesar,
Atbash, transposition, or substitution**. If it drops toward random, suspect a
**polyalphabetic** cipher like Vigenère.

## Shannon entropy

Entropy of letter frequencies, in bits per letter. English prose sits around
**4.1–4.2 bits/letter**. Higher values suggest more "mixed" output (e.g.,
Vigenère, transposition over a varied alphabet, or short noisy samples).

## Chi-squared vs English

Sum of $(O - E)^2 / E$ comparing observed letter counts to English expectations.
**Lower is more English-like.** Useful for ranking Caesar / Affine candidates.

## Caesar / ROT brute force

Only 26 shifts. Try all of them, score each against an English dictionary or
chi-squared, and the answer usually pops out.

## Atbash check

Atbash is self-inverse. Decrypting once with `A↔Z, B↔Y, ...` is a one-line test
that costs nothing.

## Kasiski / Friedman (Vigenère key length)

- **Kasiski:** repeated trigrams in the ciphertext often appear at distances
  that are multiples of the key length. The GCD of those distances suggests the
  key length.
- **Friedman:** estimate key length from the IoC. Cipher Detective AI uses a
  combined "Vigenère indicator" score derived from these.

## Transposition tells

Letter frequencies look **English** (because letters are only rearranged), but
common bigrams like `TH`, `HE`, `IN` are unusually rare. Rail-fence and
columnar transposition both produce this signature.

## Affine

Affine = $E(x) = (a x + b) \mod 26$ with $\gcd(a, 26) = 1$. Only 12 valid `a`
values × 26 `b` values = 312 keys. Brute-forceable; each candidate is scored
against English.

## Substitution

If frequency analysis matches English shape but Caesar / Atbash fail, suspect a
general monoalphabetic substitution. Solving it cleanly needs interactive
hill-climbing over an English language model — outside the scope of this demo.

## Reality check

These signals work because classical ciphers leak structure. Modern symmetric
encryption (AES-GCM, ChaCha20-Poly1305) and modern public-key cryptography do
**not** leak any of this information; none of these techniques apply to them.
