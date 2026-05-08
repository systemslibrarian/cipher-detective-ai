# Sample Ciphertexts

Use these in **Detect Mode** or **Compare Mode**. Expected answers are at the bottom — try to call them yourself first using **Explain Mode**.

## 1 — Caesar / ROT

```text
WKLV LV D FODVVLFDO FDHVDU FLSKHU GHPR IRU FLSKHU GHWHFWLYH DL
```

## 2 — Atbash

```text
GSV XLWV RH ZOO BLFIH GSV VEVIVHG RMP
```

## 3 — Vigenère (short, deliberately hard)

```text
LXFOPVEFRNHR
```

## 4 — Rail Fence

```text
TEITELHDVLSNHDTISEIIEA
```

## 5 — Columnar transposition

```text
EOACT IPTRH IIEEN HSGES SOSCR REMEN AERTC OEFNT TYIHE THCMC
```

## 6 — Plaintext

```text
THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY
```

## 7 — Affine

```text
IZZWVU NWHJUS NSV BUKUSO YUSL NWHJUSE
```

## 8 — Monoalphabetic substitution

```text
GUF KSCQNQA HQFDFQXFD ZRMVKFTBF YMQ GUF EMSSWRSGA
```

---

## Expected answers

| # | Cipher                        | Notes                                              |
|---|-------------------------------|----------------------------------------------------|
| 1 | Caesar shift 3                | Brute-forces in 26 tries. Chi-squared confirms.    |
| 2 | Atbash                        | Self-inverse — a single decode test wins.          |
| 3 | Vigenère, key `LEMON`         | Sample is too short for high confidence — that's the lesson. |
| 4 | Rail-fence (3 rails)          | English letters, weak bigrams: classic transposition signature. |
| 5 | Columnar transposition        | Same family as #4; bigram support stays low.       |
| 6 | Plaintext                     | Should classify confidently as `plaintext`.        |
| 7 | Affine, *a*=5, *b*=8          | 312-key brute force; the affine candidate table surfaces it. |
| 8 | Monoalphabetic substitution   | English-like IoC but disrupted bigrams.            |

