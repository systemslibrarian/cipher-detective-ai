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
    atbash_plain = atbash(text)

    notes: List[str] = []
    if len(letters) < 40:
        notes.append("Short samples are hard to classify reliably. Add more ciphertext for better evidence.")
    if 0.060 <= ioc <= 0.075:
        notes.append("Index of coincidence is close to natural English, which often points to monoalphabetic substitution, transposition, or plaintext.")
    elif 0.035 <= ioc <= 0.050:
        notes.append("Index of coincidence is lower than English, which can suggest polyalphabetic ciphers such as Vigenère.")
    if caesar_candidates and caesar_candidates[0][2] > 0:
        notes.append(f"Caesar shift {caesar_candidates[0][0]} produces recognizable English words.")
    if word_score(atbash_plain) > 0:
        notes.append("Atbash reversal produces recognizable English words.")
    if entropy > 4.2:
        notes.append("Entropy is relatively high for A-Z text; this may indicate stronger mixing or a short/noisy sample.")
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
        notes=notes,
    )


def heuristic_classify(text: str) -> ModelPrediction:
    letters = clean_letters(text)
    if len(letters) < 20:
        return ModelPrediction("too_short", 0.20, {"too_short": 0.20}, "heuristic")

    ev = analyze_evidence(text)
    ioc = ev.index_of_coincidence
    best_shift, best_chi, best_words, _best_plain = ev.caesar_candidates[0]
    atbash_words = word_score(ev.atbash_plaintext)
    raw_words = word_score(text)
    raw_chi = chi_squared_for_english(letters)

    scores = {
        "plaintext": 0.05,
        "caesar_rot": 0.05,
        "atbash": 0.05,
        "vigenere": 0.05,
        "transposition": 0.05,
        "substitution": 0.05,
        "affine": 0.04,
    }

    if raw_words >= 2 and raw_chi < 180:
        scores["plaintext"] += min(0.55, 0.12 * raw_words)
    if best_words >= 1:
        scores["caesar_rot"] += min(0.70, 0.20 * best_words)
    if best_shift in {13, 3, 1, 25} and best_chi < 180:
        scores["caesar_rot"] += 0.12
    if atbash_words >= 1:
        scores["atbash"] += min(0.70, 0.22 * atbash_words)
    if 0.034 <= ioc <= 0.050 and len(letters) > 60:
        scores["vigenere"] += 0.35
    if 0.055 <= ioc <= 0.080 and raw_words == 0 and best_words == 0:
        scores["substitution"] += 0.25
        scores["transposition"] += 0.18
    if ev.entropy > 4.0 and len(letters) > 50:
        scores["vigenere"] += 0.12
        scores["transposition"] += 0.08

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
