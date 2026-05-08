"""
Core cryptanalysis utilities for Cipher Detective AI.

These functions are intentionally dependency-light so they can be tested without
loading Gradio, Transformers, or a hosted model. The project combines:

1. transparent, human-readable classical cryptanalysis signals; and
2. an optional Transformer classifier for Hugging Face-native deployment.

Educational use only. This does not break modern encryption.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENGLISH_IOC = 0.0667        # English plaintext / monoalphabetic baseline.
RANDOM_IOC = 1.0 / 26.0     # ~0.0385 — uniform random letters.
ENGLISH_FREQ = {
    "A": 8.167, "B": 1.492, "C": 2.782, "D": 4.253, "E": 12.702, "F": 2.228,
    "G": 2.015, "H": 6.094, "I": 6.966, "J": 0.153, "K": 0.772, "L": 4.025,
    "M": 2.406, "N": 6.749, "O": 7.507, "P": 1.929, "Q": 0.095, "R": 5.987,
    "S": 6.327, "T": 9.056, "U": 2.758, "V": 0.978, "W": 2.360, "X": 0.150,
    "Y": 1.974, "Z": 0.074,
}
COMMON_WORDS = [
    "THE", "AND", "THAT", "HAVE", "FOR", "NOT", "WITH", "YOU", "THIS", "BUT",
    "SYSTEM", "SECURITY", "CIPHER", "MESSAGE", "MODEL", "PATTERN", "ATTACK",
    "LIBRARY", "KNOWLEDGE", "COMMUNITY", "CRYPTOGRAPHY", "FREQUENCY"
]
BIGRAMS = ["TH", "HE", "IN", "ER", "AN", "RE", "ON", "AT", "EN", "ND", "TI", "ES", "OR", "TE"]
TRIGRAMS = ["THE", "AND", "ING", "ION", "ENT", "HER", "FOR", "THA", "NTH", "INT"]

@dataclass
class ModelPrediction:
    label: str
    confidence: float
    scores: Dict[str, float]
    source: str

@dataclass
class Evidence:
    letters: int
    unique_letters: int
    index_of_coincidence: float
    entropy: float
    chi_squared: float
    top_letters: List[Tuple[str, int, float]]
    top_bigrams: List[Tuple[str, int]]
    top_trigrams: List[Tuple[str, int]]
    caesar_candidates: List[Tuple[int, float, int, str]]
    atbash_plaintext: str
    affine_candidates: List[Tuple[int, int, float, int, str]]
    friedman_key_length: float
    kasiski_key_lengths: List[Tuple[int, int]]
    transposition_signal: float
    bigram_support: float
    notes: List[str]


def clean_letters(text: str) -> str:
    return re.sub(r"[^A-Z]", "", text.upper())


def caesar_encrypt(text: str, shift: int) -> str:
    out = []
    for ch in text.upper():
        if ch in ALPHABET:
            out.append(ALPHABET[(ALPHABET.index(ch) + shift) % 26])
        else:
            out.append(ch)
    return "".join(out)


def caesar_shift(text: str, shift: int) -> str:
    """Decode by shifting letters backward."""
    out = []
    for ch in text.upper():
        if ch in ALPHABET:
            out.append(ALPHABET[(ALPHABET.index(ch) - shift) % 26])
        else:
            out.append(ch)
    return "".join(out)


def atbash(text: str) -> str:
    return text.upper().translate(str.maketrans(ALPHABET, ALPHABET[::-1]))


def affine_encrypt(text: str, a: int, b: int) -> str:
    out = []
    for ch in text.upper():
        if ch in ALPHABET:
            x = ALPHABET.index(ch)
            out.append(ALPHABET[(a * x + b) % 26])
        else:
            out.append(ch)
    return "".join(out)


def _mod_inverse(a: int, m: int = 26) -> int | None:
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None


def affine_decrypt(text: str, a: int, b: int) -> str:
    inv = _mod_inverse(a)
    if inv is None:
        raise ValueError("a must be coprime with 26")
    out = []
    for ch in text.upper():
        if ch in ALPHABET:
            y = ALPHABET.index(ch)
            out.append(ALPHABET[(inv * (y - b)) % 26])
        else:
            out.append(ch)
    return "".join(out)


def vigenere_encrypt(text: str, key: str) -> str:
    key = clean_letters(key)
    if not key:
        raise ValueError("key must contain at least one A-Z letter")
    out, j = [], 0
    for ch in text.upper():
        if ch in ALPHABET:
            k = ALPHABET.index(key[j % len(key)])
            out.append(ALPHABET[(ALPHABET.index(ch) + k) % 26])
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def vigenere_decrypt(text: str, key: str) -> str:
    """Inverse of :func:`vigenere_encrypt`."""
    key = clean_letters(key)
    if not key:
        raise ValueError("key must contain at least one A-Z letter")
    out, j = [], 0
    for ch in text.upper():
        if ch in ALPHABET:
            k = ALPHABET.index(key[j % len(key)])
            out.append(ALPHABET[(ALPHABET.index(ch) - k) % 26])
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def substitution_encrypt(text: str, mapping: str) -> str:
    """Monoalphabetic substitution. ``mapping`` is a 26-letter permutation."""
    mapping = mapping.upper()
    if len(mapping) != 26 or set(mapping) != set(ALPHABET):
        raise ValueError("mapping must be a permutation of A-Z")
    return text.upper().translate(str.maketrans(ALPHABET, mapping))


def rail_fence_encrypt(text: str, rails: int = 3) -> str:
    chars = clean_letters(text)
    if rails < 2:
        return chars
    fence = ["" for _ in range(rails)]
    rail, direction = 0, 1
    for ch in chars:
        fence[rail] += ch
        if rail == 0:
            direction = 1
        elif rail == rails - 1:
            direction = -1
        rail += direction
    return "".join(fence)


def columnar_transposition_encrypt(text: str, key: str) -> str:
    chars = clean_letters(text)
    key_letters = clean_letters(key)
    if not key_letters:
        return chars
    cols = len(key_letters)
    padding = (-len(chars)) % cols
    chars += "X" * padding
    rows = [chars[i:i+cols] for i in range(0, len(chars), cols)]
    order = sorted(range(cols), key=lambda i: (key_letters[i], i))
    return "".join("".join(row[i] for row in rows) for i in order)


def chi_squared_for_english(letters: str) -> float:
    if not letters:
        return float("inf")
    counts = Counter(letters)
    n = len(letters)
    chi = 0.0
    for ch in ALPHABET:
        observed = counts.get(ch, 0)
        expected = ENGLISH_FREQ[ch] * n / 100.0
        if expected:
            chi += ((observed - expected) ** 2) / expected
    return chi


def index_of_coincidence(letters: str) -> float:
    n = len(letters)
    if n < 2:
        return 0.0
    counts = Counter(letters)
    return sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))


def shannon_entropy(letters: str) -> float:
    if not letters:
        return 0.0
    n = len(letters)
    return -sum((c/n) * math.log2(c/n) for c in Counter(letters).values())


def word_score(text: str) -> int:
    joined = re.sub(r"[^A-Z ]", " ", text.upper())
    return sum(joined.count(w) for w in COMMON_WORDS)


def ngram_counts(letters: str, n: int, top: int = 8) -> List[Tuple[str, int]]:
    if len(letters) < n:
        return []
    c = Counter(letters[i:i+n] for i in range(len(letters) - n + 1))
    return c.most_common(top)


def best_caesar_candidates(text: str, top_n: int = 5) -> List[Tuple[int, float, int, str]]:
    candidates = []
    for shift in range(26):
        decoded = caesar_shift(text, shift)
        letters = clean_letters(decoded)
        chi = chi_squared_for_english(letters)
        score = word_score(decoded)
        # Prefer actual words, then English-like letter frequency.
        candidates.append((shift, chi, score, decoded))
    candidates.sort(key=lambda x: (-x[2], x[1]))
    return candidates[:top_n]


def best_affine_candidates(text: str, top_n: int = 5) -> List[Tuple[int, int, float, int, str]]:
    """Brute-force the 312 valid Affine keys and return the most English-looking decryptions."""
    candidates: List[Tuple[int, int, float, int, str]] = []
    for a in (1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25):
        for b in range(26):
            try:
                decoded = affine_decrypt(text, a, b)
            except ValueError:
                continue
            letters = clean_letters(decoded)
            chi = chi_squared_for_english(letters)
            score = word_score(decoded)
            candidates.append((a, b, chi, score, decoded))
    candidates.sort(key=lambda x: (-x[3], x[2]))
    return candidates[:top_n]


def friedman_key_length(letters: str) -> float:
    """Friedman estimate of Vigenère key length from the index of coincidence.

    Returns ``0.0`` if the sample is too short or the IoC is too close to random.
    """
    n = len(letters)
    if n < 40:
        return 0.0
    ioc = index_of_coincidence(letters)
    denom = (n - 1) * ioc - RANDOM_IOC * n + ENGLISH_IOC
    if denom <= 1e-6:
        return 0.0
    k = (ENGLISH_IOC - RANDOM_IOC) * n / denom
    return max(0.0, k)


def kasiski_key_lengths(letters: str, min_len: int = 3, top: int = 5) -> List[Tuple[int, int]]:
    """Kasiski examination: factor distances between repeated trigrams.

    Returns ``[(candidate_length, support_count), ...]`` sorted by support.
    """
    if len(letters) < 30:
        return []
    positions: Dict[str, List[int]] = {}
    for i in range(len(letters) - min_len + 1):
        positions.setdefault(letters[i:i + min_len], []).append(i)
    factor_support: Counter = Counter()
    for _gram, idxs in positions.items():
        if len(idxs) < 2:
            continue
        for a, b in zip(idxs, idxs[1:]):
            d = b - a
            for k in range(2, min(21, d + 1)):
                if d % k == 0:
                    factor_support[k] += 1
    return factor_support.most_common(top)


def transposition_signal(letters: str) -> Tuple[float, float]:
    """Heuristic signal for transposition ciphers.

    Transposition keeps English letter frequencies intact (so chi-squared looks
    English-like) but destroys common bigrams. Returns ``(transposition, bigram_support)``
    in ``[0, 1]``. High ``transposition`` and low ``bigram_support`` together
    suggest rail-fence / columnar.
    """
    n = len(letters)
    if n < 30:
        return 0.0, 0.0
    chi = chi_squared_for_english(letters)
    # Normalise chi against a soft cap so values >= 200 saturate to 0.
    english_likeness = max(0.0, 1.0 - min(chi, 200.0) / 200.0)
    bigrams = ngram_counts(letters, 2, top=200)
    total_bigrams = max(1, n - 1)
    common_hits = sum(
        c for bg, c in bigrams
        if bg in {"TH", "HE", "IN", "ER", "AN", "RE", "ON", "AT", "EN", "ND"}
    )
    bigram_support = min(1.0, common_hits / (total_bigrams * 0.06))  # ~6% in English
    transposition = english_likeness * (1.0 - bigram_support)
    return round(transposition, 4), round(bigram_support, 4)


def frequency_table(letters: str) -> List[Tuple[str, int, float]]:
    n = max(len(letters), 1)
    counts = Counter(letters)
    return [(ch, count, round(100 * count / n, 2)) for ch, count in counts.most_common()]



def analyze_evidence(text: str) -> Evidence:
    letters = clean_letters(text)
    top_letters = frequency_table(letters)[:10]
    ioc = index_of_coincidence(letters)
    entropy = shannon_entropy(letters)
    chi = chi_squared_for_english(letters)
    caesar_candidates = best_caesar_candidates(text, 5)
    affine_cands = best_affine_candidates(text, 5) if len(letters) >= 30 else []
    atbash_plain = atbash(text)
    friedman = round(friedman_key_length(letters), 2)
    kasiski = kasiski_key_lengths(letters)
    transp, bigram_sup = transposition_signal(letters)

    notes: List[str] = []
    if len(letters) < 40:
        notes.append("Short samples are hard to classify reliably. Add more ciphertext for better evidence.")
    if 0.060 <= ioc <= 0.075:
        notes.append("Index of coincidence is close to natural English (~0.067), which often points to plaintext, monoalphabetic substitution, or transposition.")
    elif 0.035 <= ioc <= 0.050:
        notes.append("Index of coincidence is below English, which can suggest a polyalphabetic cipher such as Vigenère.")
    if caesar_candidates and caesar_candidates[0][2] > 0:
        notes.append(f"Caesar shift {caesar_candidates[0][0]} produces recognizable English words.")
    if affine_cands and affine_cands[0][3] > 0:
        a, b, _, words, _ = affine_cands[0]
        notes.append(f"Affine key (a={a}, b={b}) produces {words} English-word match(es).")
    if word_score(atbash_plain) > 0:
        notes.append("Atbash reversal produces recognizable English words.")
    if friedman and 2 <= friedman <= 12:
        notes.append(f"Friedman estimate suggests a Vigenère-like key length near {friedman:.1f}.")
    if kasiski:
        top_k = ", ".join(f"{k} (×{n})" for k, n in kasiski[:3])
        notes.append(f"Kasiski-style repeated trigrams support key length(s): {top_k}.")
    if transp >= 0.55 and bigram_sup <= 0.35:
        notes.append("Letter frequencies look English but common bigrams (TH, HE, IN…) are scarce — classic transposition signature.")
    if entropy > 4.2:
        notes.append("Entropy is relatively high for A–Z text; this may indicate stronger mixing or a short/noisy sample.")
    if len(set(letters)) < 10 and len(letters) > 20:
        notes.append("Very few unique letters; this sample may be too constrained or not normal prose.")

    return Evidence(
        letters=len(letters),
        unique_letters=len(set(letters)),
        index_of_coincidence=round(ioc, 5),
        entropy=round(entropy, 4),
        chi_squared=round(chi, 2) if math.isfinite(chi) else float("inf"),
        top_letters=top_letters,
        top_bigrams=ngram_counts(letters, 2),
        top_trigrams=ngram_counts(letters, 3),
        caesar_candidates=caesar_candidates,
        atbash_plaintext=atbash_plain,
        affine_candidates=affine_cands,
        friedman_key_length=friedman,
        kasiski_key_lengths=kasiski,
        transposition_signal=transp,
        bigram_support=bigram_sup,
        notes=notes,
    )


def heuristic_classify(text: str) -> ModelPrediction:
    """Transparent heuristic classifier.

    Returns a normalized score per label. The label set matches the dataset:
    ``plaintext, caesar_rot, atbash, vigenere, rail_fence, columnar, affine, substitution``.
    """
    letters = clean_letters(text)
    if len(letters) < 20:
        return ModelPrediction(
            "too_short",
            0.20,
            {"too_short": 0.20},
            "heuristic",
        )

    ev = analyze_evidence(text)
    ioc = ev.index_of_coincidence
    best_shift, best_chi, best_words, _ = ev.caesar_candidates[0]
    atbash_words = word_score(ev.atbash_plaintext)
    affine_words = ev.affine_candidates[0][3] if ev.affine_candidates else 0
    raw_words = word_score(text)
    raw_chi = chi_squared_for_english(letters)

    scores: Dict[str, float] = {
        "plaintext":   0.04,
        "caesar_rot":  0.04,
        "atbash":      0.04,
        "vigenere":    0.04,
        "rail_fence":  0.04,
        "columnar":    0.04,
        "substitution": 0.04,
        "affine":      0.04,
    }

    # --- Plaintext ----------------------------------------------------------
    if raw_words >= 2 and raw_chi < 180:
        scores["plaintext"] += min(0.65, 0.14 * raw_words)

    # --- Caesar -------------------------------------------------------------
    if best_words >= 1:
        scores["caesar_rot"] += min(0.70, 0.22 * best_words)
    if best_shift in {1, 3, 13, 25} and best_chi < 180:
        scores["caesar_rot"] += 0.12

    # --- Atbash -------------------------------------------------------------
    if atbash_words >= 1:
        scores["atbash"] += min(0.75, 0.24 * atbash_words)

    # --- Affine -------------------------------------------------------------
    # Only fire if the affine guess is *better* than the best Caesar guess —
    # otherwise the same word matches will inflate both labels.
    if affine_words >= max(2, best_words + 1):
        scores["affine"] += min(0.65, 0.18 * affine_words)

    # --- Vigenère -----------------------------------------------------------
    fried = ev.friedman_key_length
    kas_support = sum(n for _, n in ev.kasiski_key_lengths[:3])
    if 0.034 <= ioc <= 0.050 and len(letters) > 60:
        scores["vigenere"] += 0.30
    if 2 <= fried <= 12:
        scores["vigenere"] += 0.20
    if kas_support >= 3:
        scores["vigenere"] += min(0.20, 0.04 * kas_support)
    if ev.entropy > 4.0 and len(letters) > 50 and raw_words == 0 and best_words == 0:
        scores["vigenere"] += 0.08

    # --- Transposition (rail_fence + columnar share the signal) -------------
    if ev.transposition_signal >= 0.50 and ev.bigram_support <= 0.40:
        # Split the bonus; columnar tends to leave longer English-like runs,
        # rail-fence shorter ones. Without a key we can't be certain — give
        # rail_fence a slight edge on shorter samples.
        bonus = min(0.55, 0.7 * ev.transposition_signal)
        if len(letters) <= 80:
            scores["rail_fence"] += bonus
            scores["columnar"]   += bonus * 0.7
        else:
            scores["columnar"]   += bonus
            scores["rail_fence"] += bonus * 0.7

    # --- Substitution -------------------------------------------------------
    # English-like IoC, no Caesar/Atbash/Affine match, and bigrams *also*
    # don't look English (because the substitution permutes them).
    if (
        0.055 <= ioc <= 0.080
        and raw_words == 0
        and best_words == 0
        and atbash_words == 0
        and affine_words == 0
        and ev.bigram_support <= 0.55
    ):
        scores["substitution"] += 0.45

    total = sum(scores.values())
    norm = {k: round(v / total, 4) for k, v in scores.items()}
    label = max(norm, key=norm.get)
    return ModelPrediction(label, norm[label], norm, "heuristic")



def build_explanation(text: str, pred: ModelPrediction) -> str:
    ev = analyze_evidence(text)
    lines = [
        f"### Detective conclusion: `{pred.label}`",
        f"**Confidence:** {pred.confidence:.1%}  ",
        f"**Prediction source:** {pred.source}",
        "",
        "### Evidence",
        f"- Letters analyzed: **{ev.letters}**",
        f"- Unique A-Z letters: **{ev.unique_letters}**",
        f"- Index of coincidence: **{ev.index_of_coincidence}**",
        f"- Shannon entropy: **{ev.entropy}** bits/letter",
        f"- English chi-squared score: **{ev.chi_squared}**",
        "",
        "### Why that matters",
    ]
    if ev.notes:
        lines.extend([f"- {note}" for note in ev.notes])
    else:
        lines.append("- No single signal dominates. Treat this as a hypothesis, not a proof.")

    lines += [
        "",
        "### Top Caesar / ROT candidates",
        "| Shift | Word clues | Chi-squared | Preview |",
        "|---:|---:|---:|---|",
    ]
    for shift, chi, score, decoded in ev.caesar_candidates[:5]:
        preview = decoded[:110].replace("\n", " ")
        lines.append(f"| {shift} | {score} | {chi:.2f} | `{preview}` |")

    if ev.affine_candidates:
        lines += [
            "",
            "### Top Affine candidates",
            "| a | b | Word clues | Chi-squared | Preview |",
            "|---:|---:|---:|---:|---|",
        ]
        for a, b, chi, score, decoded in ev.affine_candidates[:3]:
            preview = decoded[:90].replace("\n", " ")
            lines.append(f"| {a} | {b} | {score} | {chi:.2f} | `{preview}` |")

    lines += [
        "",
        "### Polyalphabetic indicators",
        f"- Friedman key-length estimate: **{ev.friedman_key_length or '—'}**",
        "- Kasiski candidate key lengths: "
        + (", ".join(f"`{k}` (×{n})" for k, n in ev.kasiski_key_lengths[:5]) or "—"),
        "",
        "### Transposition indicators",
        f"- Transposition signal: **{ev.transposition_signal}** (high = English letters but disrupted bigrams)",
        f"- English-bigram support: **{ev.bigram_support}**",
    ]

    lines += [
        "",
        "### Top letter frequencies",
        "| Letter | Count | Percent |",
        "|---|---:|---:|",
    ]
    for ch, count, pct in ev.top_letters[:10]:
        lines.append(f"| {ch} | {count} | {pct}% |")

    lines += [
        "",
        "### Reality check",
        "This project teaches classical cryptanalysis signals. It does **not** break modern encryption, recover passwords, bypass access controls, or prove anything about real-world cryptographic security.",
    ]
    return "\n".join(lines)

