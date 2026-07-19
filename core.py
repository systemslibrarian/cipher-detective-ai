"""
Core cryptanalysis utilities for Cipher Detective AI.

These functions are intentionally dependency-light so they can be tested without
loading Gradio, Transformers, or a hosted model. The project combines:

1. transparent, human-readable classical cryptanalysis signals; and
2. an optional Transformer classifier for Hugging Face-native deployment.

Educational use only. This does not break modern encryption.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ngram_tables import _BIGRAM_LOG_PROB, _QUADGRAM_LOG_PROB, _TRIGRAM_LOG_PROB

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
    # High-frequency English function/content words (4+ chars — substring match is safe)
    "THAT", "HAVE", "WITH", "FROM", "WILL", "THEY", "BEEN", "WHEN",
    "WERE", "INTO", "THAN", "THEM", "MORE", "WHAT", "YOUR", "EACH", "SOME",
    "OVER", "JUST", "ALSO", "BACK", "VERY", "TIME", "COME", "MAKE", "TAKE",
    "ONLY", "EVEN", "KNOW", "MANY", "MOST", "SUCH", "MUST", "LONG", "UPON",
    "UNTO", "SEND", "GIVE", "FIND", "WELL", "PART", "HAND", "HERE", "SAID",
    "LAST", "NEXT", "BOTH", "LIFE", "DEATH", "LOVE", "KING", "LORD", "ARMY",
    "ENEMY", "SECRET", "CIPHER", "CIPHERS", "MESSAGE", "ATTACK", "LETTER",
    "LETTERS", "BEFORE", "SHOULD", "FORCE", "ORDER", "PAIRS", "THREE",
    "HUMAN", "STRONG", "SECURE", "PATTERN", "KEYWORD", "MACHINE", "WITHOUT",
    "EVIDENCE", "ALPHABET", "ENCODING", "FREQUENCY", "KNOWLEDGE", "ENCRYPTION",
    "TRANSPOSITION", "SUBSTITUTION", "COINCIDENCE", "POLYALPHABETIC",
    "CIPHERTEXT", "PLAINTEXT", "POLYBIUS", "VIGENERE", "ENIGMA", "COLUMNAR",
    "INDEX", "PUBLIC", "SIGNAL", "SQUARE", "ITSELF", "CAREFUL", "BRUTE",
    "SHOW", "USES", "USED", "WORKS", "CLAIM", "NINETEEN",
    # Short words matched as whole words only (stored separately, see word_score)
    "THE", "AND", "FOR", "NOT", "YOU", "THIS", "BUT", "ARE", "CAN", "HIS",
    "HER", "OUR", "OUT", "WHO", "ITS", "WAS", "HAS", "NOW", "HIM", "MAY",
    "SAY", "ALL", "ONE", "GET", "GOD", "MAN", "WAR", "END", "DAY", "KEY",
    "USE", "TWO",
]
# Short words (≤3 chars) need whole-word matching to avoid false positives.
# Substring matching of "THE" across word boundaries in cipher text causes
# ~40% of short texts to spuriously score ≥ 1, breaking the brute-force gate.
_SHORT_WORDS = frozenset(w for w in COMMON_WORDS if len(w) <= 3)
BIGRAMS = ["TH", "HE", "IN", "ER", "AN", "RE", "ON", "AT", "EN", "ND", "TI", "ES", "OR", "TE"]
TRIGRAMS = ["THE", "AND", "ING", "ION", "ENT", "HER", "FOR", "THA", "NTH", "INT"]

# Common historical Vigenère keywords, tried before the statistical attack.
# On short ciphertexts (too few letters per column for chi-squared to grip) a
# known keyword recovers the plaintext where column analysis cannot.
COMMON_KEYWORDS = [
    "KEY", "CODE", "SECRET", "CIPHER", "PASSWORD", "ENIGMA", "SPHINX", "RIDDLE",
    "SHADOW", "VICTORY", "FORTRESS", "CASTLE", "DRAGON", "PHANTOM", "LIBERTY",
    "FREEDOM", "JUSTICE", "EMPIRE", "LEMON", "MELON", "MUSEUM", "LIBRARY",
    "SIGNAL", "AUTUMN", "WINTER", "SUMMER", "SPRING", "THUNDER", "STORM",
    "EAGLE", "FALCON", "RAVEN", "TIGER", "COBRA", "VIPER", "ORACLE", "MATRIX",
    "GENERAL", "COLONEL", "CAPTAIN", "SOLDIER", "BATTLE", "NELSON", "CAESAR",
    "NAPOLEON", "WELLINGTON", "GOLD", "SILVER", "DIAMOND", "CRYSTAL",
]

@dataclass
class ModelPrediction:
    label: str
    confidence: float
    scores: dict[str, float]
    source: str

@dataclass
class Evidence:
    letters: int
    unique_letters: int
    index_of_coincidence: float
    entropy: float
    chi_squared: float
    top_letters: list[tuple[str, int, float]]
    top_bigrams: list[tuple[str, int]]
    top_trigrams: list[tuple[str, int]]
    caesar_candidates: list[tuple[int, float, int, str]]
    atbash_plaintext: str
    affine_candidates: list[tuple[int, int, float, int, str]]
    friedman_key_length: float
    kasiski_key_lengths: list[tuple[int, int]]
    transposition_signal: float
    bigram_support: float
    notes: list[str]


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


def beaufort_decrypt(text: str, key: str) -> str:
    """Beaufort cipher decryption.  Beaufort is reciprocal: c = (k - p) mod 26,
    so decryption uses the same operation: p = (k - c) mod 26."""
    key = clean_letters(key)
    if not key:
        raise ValueError("key must contain at least one A-Z letter")
    out, j = [], 0
    for ch in text.upper():
        if ch in ALPHABET:
            k = ALPHABET.index(key[j % len(key)])
            c = ALPHABET.index(ch)
            out.append(ALPHABET[(k - c) % 26])
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
    tokens = set(joined.split())
    # Long words: substring count (safe); short words: whole-word match only
    score = sum(joined.count(w) for w in COMMON_WORDS if w not in _SHORT_WORDS)
    score += sum(1 for w in _SHORT_WORDS if w in tokens)
    return score


def ngram_counts(letters: str, n: int, top: int = 8) -> list[tuple[str, int]]:
    if len(letters) < n:
        return []
    c = Counter(letters[i:i+n] for i in range(len(letters) - n + 1))
    return c.most_common(top)


def best_caesar_candidates(text: str, top_n: int = 5) -> list[tuple[int, float, int, str]]:
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


def best_affine_candidates(text: str, top_n: int = 5) -> list[tuple[int, int, float, int, str]]:
    """Brute-force the 312 valid Affine keys and return the most English-looking decryptions."""
    candidates: list[tuple[int, int, float, int, str]] = []
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
    # Rank by word hits, then bigram log-prob: on short texts chi-squared is
    # too noisy to pick between near-tied keys, but letter-order structure
    # (bigrams) still discriminates well.
    candidates.sort(key=lambda x: (-x[3], -english_bigram_score(clean_letters(x[4])), x[2]))
    return candidates[:top_n]


def vigenere_auto_solve(
    ciphertext: str,
    max_key_len: int = 15,
    top_n: int = 5,
) -> list[tuple[str, str, float, int]]:
    """Auto-solve a Vigenère cipher using Kasiski / Friedman key-length estimation
    followed by per-column Caesar brute-force.

    Returns up to ``top_n`` candidates as ``(key, plaintext, chi_sq, word_count)``
    sorted best-first (most English words, then lowest chi-squared).
    """
    letters = clean_letters(ciphertext)
    if len(letters) < 20:
        return []

    # Gather candidate key lengths from Kasiski + Friedman
    kasiski = kasiski_key_lengths(letters, top=5)
    fried = friedman_key_length(letters)

    key_len_candidates: set[int] = set()
    for k, _ in kasiski[:4]:
        if 2 <= k <= max_key_len:
            key_len_candidates.add(k)
    if fried:
        for f in {int(fried), round(int(fried + 0.5))}:  # floor and nearest int
            if 2 <= f <= max_key_len:
                key_len_candidates.add(f)
    # Always try short lengths as a safety net
    for k in range(2, min(9, max_key_len + 1)):
        key_len_candidates.add(k)

    results: list[tuple[str, str, float, int]] = []
    seen_keys: set[str] = set()

    for key_len in sorted(key_len_candidates):
        # Split ciphertext letters into key_len interleaved streams
        streams = [letters[i::key_len] for i in range(key_len)]
        key_letters = []
        for stream in streams:
            # Find the Caesar shift (= key letter position) that minimises chi-squared
            best_chi_col = float("inf")
            best_s = 0
            for s in range(26):
                decoded_col = "".join(
                    ALPHABET[(ALPHABET.index(c) - s) % 26] for c in stream
                )
                chi_col = chi_squared_for_english(decoded_col)
                if chi_col < best_chi_col:
                    best_chi_col = chi_col
                    best_s = s
            key_letters.append(ALPHABET[best_s])

        # Refinement: per-column chi is noisy when each column has few letters
        # (short text and/or long key).  Coordinate descent on the FULL-text
        # bigram score fixes the columns chi got wrong: hold the rest of the
        # key fixed and try all 26 letters for each position until stable.
        def _bg(kl: list[str], _klen: int = key_len) -> float:
            return english_bigram_score(
                "".join(
                    ALPHABET[(ALPHABET.index(c) - ALPHABET.index(kl[i % _klen])) % 26]
                    for i, c in enumerate(letters)
                )
            )

        best_bg = _bg(key_letters)
        for _ in range(3):  # usually converges in 1-2 sweeps
            improved = False
            for i in range(key_len):
                original = key_letters[i]
                for cand in ALPHABET:
                    if cand == original:
                        continue
                    key_letters[i] = cand
                    bg = _bg(key_letters)
                    if bg > best_bg:
                        best_bg = bg
                        original = cand
                        improved = True
                key_letters[i] = original
            if not improved:
                break

        key = "".join(key_letters)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        plaintext = vigenere_decrypt(ciphertext, key)
        plain_letters = clean_letters(plaintext)
        chi = chi_squared_for_english(plain_letters)
        ws = word_score(plaintext)
        results.append((key, plaintext, chi, ws, english_bigram_score(plain_letters)))

    # Dictionary pass: historical Vigenère keys were words. On short texts the
    # statistical attack has too few letters per column, but a common keyword
    # decodes cleanly — this is how real cryptanalysts often started.
    for key in COMMON_KEYWORDS:
        if key in seen_keys or len(key) > max_key_len:
            continue
        seen_keys.add(key)
        plaintext = vigenere_decrypt(ciphertext, key)
        plain_letters = clean_letters(plaintext)
        results.append((key, plaintext, chi_squared_for_english(plain_letters),
                        word_score(plaintext), english_bigram_score(plain_letters)))

    # Word hits first, then bigram structure (chi is too noisy on short text).
    results.sort(key=lambda x: (-x[3], -x[4], x[2]))
    return [(k, p, c, w) for k, p, c, w, _ in results[:top_n]]


def beaufort_auto_solve(
    ciphertext: str,
    max_key_len: int = 15,
    top_n: int = 5,
) -> list[tuple[str, str, float, int]]:
    """Auto-solve a Beaufort cipher (reciprocal Vigenère variant).

    Same Kasiski/Friedman key-length search as Vigenère, but each column is a
    Beaufort cipher ``p = (k - c) mod 26``, so the key letter is chosen to
    minimise chi-squared under that operation. Returns the same
    ``(key, plaintext, chi_sq, word_count)`` shape.
    """
    letters = clean_letters(ciphertext)
    if len(letters) < 20:
        return []

    kasiski = kasiski_key_lengths(letters, top=5)
    fried = friedman_key_length(letters)
    key_lens: set[int] = {k for k, _ in kasiski[:4] if 2 <= k <= max_key_len}
    if fried:
        key_lens |= {k for k in (int(fried), round(fried)) if 2 <= k <= max_key_len}
    key_lens |= set(range(2, min(9, max_key_len + 1)))

    def _decrypt_letters(kl: list[str], klen: int) -> str:
        return "".join(
            ALPHABET[(ALPHABET.index(kl[i % klen]) - ALPHABET.index(c)) % 26]
            for i, c in enumerate(letters)
        )

    results: list[tuple[str, str, float, int, float]] = []
    seen: set[str] = set()
    for key_len in sorted(key_lens):
        key_chars = []
        for col in range(key_len):
            stream = letters[col::key_len]
            best_chi, best_k = float("inf"), 0
            for s in range(26):
                dec = "".join(ALPHABET[(s - ALPHABET.index(c)) % 26] for c in stream)
                chi = chi_squared_for_english(dec)
                if chi < best_chi:
                    best_chi, best_k = chi, s
            key_chars.append(ALPHABET[best_k])

        # Coordinate descent on full-text bigram score: fixes key letters that
        # per-column chi got wrong when columns are short (same refinement as
        # the Vigenère solver).
        best_bg = english_bigram_score(_decrypt_letters(key_chars, key_len))
        for _ in range(3):
            improved = False
            for i in range(key_len):
                original = key_chars[i]
                for cand in ALPHABET:
                    if cand == original:
                        continue
                    key_chars[i] = cand
                    bg = english_bigram_score(_decrypt_letters(key_chars, key_len))
                    if bg > best_bg:
                        best_bg, original, improved = bg, cand, True
                key_chars[i] = original
            if not improved:
                break

        key = "".join(key_chars)
        if key in seen:
            continue
        seen.add(key)
        plaintext = beaufort_decrypt(ciphertext, key)
        pl = clean_letters(plaintext)
        results.append((key, plaintext, chi_squared_for_english(pl),
                        word_score(plaintext), english_bigram_score(pl)))

    results.sort(key=lambda x: (-x[3], -x[4], x[2]))
    return [(k, p, c, w) for k, p, c, w, _ in results[:top_n]]


def columnar_auto_solve(
    ciphertext: str,
    max_cols: int = 8,
    top_n: int = 5,
    seed: int | None = 42,
) -> list[tuple[str, str, float, int]]:
    """Auto-solve a columnar transposition without the key.

    Only the column *ordering* is secret (the letters are unchanged), so for
    each candidate column count we hill-climb the permutation — swapping pairs
    of columns and keeping swaps that raise the quadgram score. Returns
    ``(recovered_key_order, plaintext, quadgram_score, word_count)`` where the
    "key order" is a digit string giving the read order of columns.
    """
    import random as _random

    letters = clean_letters(ciphertext)
    n = len(letters)
    if n < 12:
        return []
    rng = _random.Random(seed)

    def _decode(cols: int, order: list[int]) -> str:
        # `order` is the read order used at encryption time (which original
        # column each ciphertext segment belongs to). Rebuild rows from it.
        full_rows, remainder = divmod(n, cols)
        col_len = [full_rows + (1 if c < remainder else 0) for c in range(cols)]
        columns = [""] * cols
        pos = 0
        for orig_col in order:
            columns[orig_col] = letters[pos:pos + col_len[orig_col]]
            pos += col_len[orig_col]
        out = []
        ptr = [0] * cols
        for _ in range(full_rows + (1 if remainder else 0)):
            for c in range(cols):
                if ptr[c] < len(columns[c]):
                    out.append(columns[c][ptr[c]])
                    ptr[c] += 1
        return "".join(out)

    results: list[tuple[str, str, float, int]] = []
    for cols in range(2, max_cols + 1):
        if cols > n:
            break
        best_order = list(range(cols))
        best_score = english_quadgram_score(_decode(cols, best_order))
        for restart in range(6):
            order = list(range(cols))
            if restart:
                rng.shuffle(order)
            score = english_quadgram_score(_decode(cols, order))
            improved = True
            while improved:
                improved = False
                for i in range(cols):
                    for j in range(i + 1, cols):
                        order[i], order[j] = order[j], order[i]
                        s = english_quadgram_score(_decode(cols, order))
                        if s > score:
                            score, improved = s, True
                        else:
                            order[i], order[j] = order[j], order[i]
            if score > best_score:
                best_score, best_order = score, order[:]
        plaintext = _decode(cols, best_order)
        results.append(("".join(str(c) for c in best_order), plaintext,
                        round(best_score, 3), word_score(plaintext)))

    results.sort(key=lambda x: (-x[3], -x[2]))
    return results[:top_n]


def playfair_double_score(letters: str) -> float:
    """Return fraction of consecutive letter-pairs that are identical.

    Playfair forbids encoding a pair with both letters the same (they would
    be in the same row/column), so a genuine Playfair ciphertext has very
    few or no identical consecutive-pair doubles.  English plaintext has
    ~3.9% (1/26), Playfair should be ≈ 0%.
    """
    n = len(letters)
    if n < 4:
        return 0.0
    pairs = [letters[i:i + 2] for i in range(0, n - 1, 2)]
    doubles = sum(1 for p in pairs if len(p) == 2 and p[0] == p[1])
    return doubles / max(len(pairs), 1)


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


def kasiski_key_lengths(letters: str, min_len: int = 3, top: int = 5) -> list[tuple[int, int]]:
    """Kasiski examination: factor distances between repeated trigrams.

    Returns ``[(candidate_length, support_count), ...]`` sorted by support.
    """
    if len(letters) < 30:
        return []
    positions: dict[str, list[int]] = {}
    for i in range(len(letters) - min_len + 1):
        positions.setdefault(letters[i:i + min_len], []).append(i)
    factor_support: Counter = Counter()
    for _gram, idxs in positions.items():
        if len(idxs) < 2:
            continue
        for a, b in zip(idxs, idxs[1:], strict=False):
            d = b - a
            for k in range(2, min(21, d + 1)):
                if d % k == 0:
                    factor_support[k] += 1
    return factor_support.most_common(top)


def transposition_signal(letters: str) -> tuple[float, float]:
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
    # Top-10 English bigrams (TH, HE, IN, …) are ~12–15% of all bigrams in real
    # text. The previous 0.06 constant saturated the ratio on any vaguely-English
    # input — including most columnar transpositions — and zeroed out the signal.
    bigram_support = min(1.0, common_hits / (total_bigrams * 0.12))
    transposition = english_likeness * (1.0 - bigram_support)
    return round(transposition, 4), round(bigram_support, 4)


# ---------------------------------------------------------------------------
# Hill-climbing solver for monoalphabetic substitution
# ---------------------------------------------------------------------------

# The n-gram scoring tables live in ngram_tables.py (imported at the top of
# this module): all 676 bigrams + the top-1000 trigrams, derived from the
# corpus plaintexts.  The previous hand-typed ~100-bigram / 60-trigram tables
# were too sparse to climb on: their unigram backoff dominated the score and
# REWARDED frequency-correct letter salad, so the hill climber's
# frequency-seeded start was already a local optimum of its own metric.
# Regenerate with: python scripts/build_ngram_tables.py > ngram_tables.py


def english_trigram_score(letters: str) -> float:
    """Mean log10-probability per trigram. Higher (less negative) = more English.

    Unseen trigrams back off to the bigram chain rule
    ``p(abc) ~ p(ab) * p(c|b) = p(ab) * p(bc) / p(b)`` — smooth everywhere
    (the bigram table is complete) and, unlike a unigram-sum backoff, it does
    not reward frequency-correct letter salad.
    """
    if len(letters) < 3:
        return -8.0
    total = 0.0
    trips = len(letters) - 2
    for i in range(trips):
        tg = letters[i:i + 3]
        seen = _TRIGRAM_LOG_PROB.get(tg)
        if seen is not None:
            total += seen
        else:
            total += (
                _BIGRAM_LOG_PROB.get(tg[:2], -6.0)
                + _BIGRAM_LOG_PROB.get(tg[1:], -6.0)
                - _UNIGRAM_LOG_PROB.get(tg[1], -3.0)
                - 0.3  # small penalty: an unseen trigram is rarer than its chain estimate
            )
    return total / trips


# log10 of English letter frequency / 100 — used as a backoff for unseen bigrams.
_UNIGRAM_LOG_PROB: dict[str, float] = {
    ch: math.log10(max(freq, 0.01) / 100.0) for ch, freq in ENGLISH_FREQ.items()
}
_BIGRAM_FLOOR = -8.0
def english_quadgram_score(letters: str) -> float:
    """Mean log10-probability per quadgram — the sharpest English discriminator.

    Quadgrams punish wrong decodes far harder than bigrams/trigrams, which
    deepens the true key's basin for the hill climber.  Unseen quadgrams back
    off to the trigram chain rule ``p(abcd) ~ p(abc) * p(bcd) / p(bc)``.
    """
    if len(letters) < 4:
        return english_trigram_score(letters)
    total = 0.0
    quads = len(letters) - 3
    for i in range(quads):
        qg = letters[i:i + 4]
        seen = _QUADGRAM_LOG_PROB.get(qg)
        if seen is not None:
            total += seen
        else:
            total += (
                _TRIGRAM_LOG_PROB.get(qg[:3], -6.5)
                + _TRIGRAM_LOG_PROB.get(qg[1:], -6.5)
                - _BIGRAM_LOG_PROB.get(qg[1:3], -6.0)
                - 0.3
            )
    return total / quads


def english_bigram_score(letters: str) -> float:
    """Mean log-probability per bigram. Higher (less negative) = more English-like.

    The bigram table covers all 676 A-Z pairs (add-one smoothed), so the
    gradient is smooth everywhere; the floor only fires for non-alphabetic
    characters that slipped past cleaning.
    """
    if len(letters) < 2:
        return _BIGRAM_FLOOR
    total = 0.0
    pairs = len(letters) - 1
    for i in range(pairs):
        total += _BIGRAM_LOG_PROB.get(letters[i:i + 2], -6.0)
    return total / pairs


def _apply_key(letters: str, key: str) -> str:
    """Apply a 26-letter substitution key (cipher A..Z -> plaintext)."""
    return letters.translate(str.maketrans(ALPHABET, key))


def hill_climb_substitution(
    text: str,
    iterations: int = 4000,
    restarts: int = 8,
    seed: int | None = 42,
) -> tuple[str, str, float]:
    """Solve monoalphabetic substitution by greedy descent on a blended
    bigram/trigram/quadgram log-probability score, with random restarts.

    Returns ``(plaintext_guess, key, score)``. ``key`` is a 26-letter string
    where ``key[i]`` is the plaintext letter for cipher letter ``ALPHABET[i]``.
    ``iterations`` is kept for API compatibility; each restart now runs full
    325-swap sweeps to a local optimum, so ``restarts`` is the effort knob.

    Educational only — reliably solves ~200+ letters of natural English,
    degrades on shorter or highly repetitive samples. That failure mode is
    part of the lesson.
    """
    import random as _random
    rng = _random.Random(seed)
    letters = clean_letters(text)
    if len(letters) < 30:
        return text, ALPHABET, _BIGRAM_FLOOR

    # Seed the key from observed letter frequency mapped to English ranking —
    # this gives the climber a head start over a random permutation.
    english_order = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
    observed = [ch for ch, _ in Counter(letters).most_common()]
    observed += [c for c in ALPHABET if c not in observed]  # fill missing letters
    seed_key = list(ALPHABET)
    for cipher_letter, plain_letter in zip(observed, english_order, strict=False):
        seed_key[ALPHABET.index(cipher_letter)] = plain_letter

    def _score(k: str) -> float:
        dec = _apply_key(letters, k)
        # Quadgrams dominate the blend: they punish wrong decodes hardest,
        # which deepens the true key's basin.  Bigrams/trigrams keep the
        # gradient smooth where quadgram hits are sparse.
        bg = english_bigram_score(dec)
        if len(dec) >= 30:
            tg = english_trigram_score(dec)
            qg = english_quadgram_score(dec)
            return 0.2 * bg + 0.3 * tg + 0.5 * qg
        return bg

    best_key = "".join(seed_key)
    best_score = _score(best_key)

    def _greedy_descent(current: list[str]) -> tuple[list[str], float]:
        """Jakobsen-style descent: sweep ALL 325 pair swaps, keep improvements,
        repeat until a full sweep finds none.  Deterministic exploitation beats
        random swap sampling, which routinely misses the one improving move."""
        current_score = _score("".join(current))
        improved = True
        while improved:
            improved = False
            for a in range(25):
                for b in range(a + 1, 26):
                    current[a], current[b] = current[b], current[a]
                    score = _score("".join(current))
                    if score > current_score:
                        current_score = score
                        improved = True
                    else:
                        current[a], current[b] = current[b], current[a]  # revert
        return current, current_score

    # Greedy descent from the frequency seed plus random restarts.  The
    # quadgram-heavy score makes the true key's basin deep enough that a
    # modest number of restarts reaches it reliably (verified empirically:
    # the truth is the best local optimum of this score).
    for restart in range(max(1, restarts)):
        if restart == 0:
            current = list(seed_key)
        else:
            current = list(ALPHABET)
            rng.shuffle(current)
        current, current_score = _greedy_descent(current)
        if current_score > best_score:
            best_score = current_score
            best_key = "".join(current)

    plaintext = _apply_key(text.upper(), best_key)
    return plaintext, best_key, best_score


def rail_fence_decrypt(text: str, rails: int) -> str:
    """Reverse rail-fence transposition for a given number of rails."""
    letters = clean_letters(text)
    n = len(letters)
    if rails < 2 or n == 0:
        return letters
    # Determine which rail each position belongs to
    pattern = []
    rail, direction = 0, 1
    for _ in range(n):
        pattern.append(rail)
        if rail == 0:
            direction = 1
        elif rail == rails - 1:
            direction = -1
        rail += direction
    # Build index map: original position → ciphertext read order
    indices = sorted(range(n), key=lambda i: (pattern[i], i))
    result = [""] * n
    for ct_pos, orig_pos in enumerate(indices):
        result[orig_pos] = letters[ct_pos]
    return "".join(result)


def best_rail_fence_candidates(text: str, max_rails: int = 15) -> list[tuple[int, str, float, int]]:
    """Brute-force rail counts 2..max_rails, return best decodes sorted by English quality."""
    candidates = []
    for r in range(2, max_rails + 1):
        decoded = rail_fence_decrypt(text, r)
        chi = chi_squared_for_english(decoded)
        ws = word_score(decoded)
        candidates.append((r, decoded, chi, ws))
    candidates.sort(key=lambda x: (-x[3], x[2]))
    return candidates[:5]


def columnar_transposition_decrypt(text: str, key: str) -> str:
    """Reverse columnar transposition given a known keyword."""
    letters = clean_letters(text)
    key_letters = clean_letters(key)
    if not key_letters or not letters:
        return letters
    cols = len(key_letters)
    n = len(letters)
    full_rows, remainder = divmod(n, cols)
    order = sorted(range(cols), key=lambda i: (key_letters[i], i))
    # Column length depends on ORIGINAL column position: when the last row is
    # partial, the leftmost `remainder` columns of the grid hold one extra char.
    col_len = [full_rows + (1 if c < remainder else 0) for c in range(cols)]
    # Ciphertext segments appear in key-sorted order (the order encrypt wrote
    # them), so walk `order` to slice each original column back out.
    columns = [""] * cols
    pos = 0
    for orig_col in order:
        columns[orig_col] = letters[pos:pos + col_len[orig_col]]
        pos += col_len[orig_col]
    # Reconstruct row by row
    result = []
    col_ptrs = [0] * cols
    for _ in range(full_rows + (1 if remainder else 0)):
        for c in range(cols):
            if col_ptrs[c] < len(columns[c]):
                result.append(columns[c][col_ptrs[c]])
                col_ptrs[c] += 1
    return "".join(result)


def frequency_table(letters: str) -> list[tuple[str, int, float]]:
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

    notes: list[str] = []
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


# ---------------------------------------------------------------------------
# All 81 cipher labels present in the full dataset.
# ---------------------------------------------------------------------------
_ALL_LABELS: list[str] = [
    "adfgvx", "adfgx", "aeneas_tacticus", "affine", "alberti_disk", "argenti",
    "arnold_andre", "atbash", "autokey", "babington", "bacon_cipher", "bazeries",
    "beaufort", "bifid", "book_cipher", "caesar", "caesar_rot", "cardano_autokey",
    "chaocipher", "chinese_telegraph", "columnar", "columnar_transposition",
    "commercial_code", "confederate_vigenere", "copiale", "culper_ring", "diana",
    "double_transposition", "enigma", "fialka", "four_square", "fractionated_morse",
    "geez_monastic", "geheimschreiber", "great_cipher", "gronsfeld", "hill",
    "homophonic", "jefferson_disk", "jn25", "joseon_yeokhak", "kama_sutra", "kl7",
    "kryha", "kryptos", "lorenz", "m209", "m94", "monoalphabetic", "morse_code",
    "navajo_code", "nihilist", "nomenclator", "null_cipher", "one_time_pad",
    "pigpen", "plaintext", "playfair", "polybius", "porta", "purple", "rail_fence",
    "red_type_a", "rot13", "running_key", "scytale", "sigaba", "slidex",
    "solitaire", "stager_route", "straddling_checkerboard", "substitution",
    "tap_code", "trifid", "trithemius", "two_square", "typex", "venona_pad_reuse",
    "vernam", "vic", "vigenere", "voynich_render", "wallis_cipher", "wheatstone",
    "zimmermann",
]

# ---------------------------------------------------------------------------
# Cipher families: groups of labels that share a statistical signature.
#
# Many fine-grained labels are *mathematically indistinguishable* from
# ciphertext alone (e.g. every well-built rotor machine emits a near-uniform
# letter stream; kama_sutra IS a monoalphabetic substitution).  Family-level
# accuracy is therefore the honest headline metric; fine-grained labels are
# best-effort within a family.  Assignments follow how each cipher *renders in
# this corpus*, which occasionally differs from the historical device (the
# corpus renders lorenz as lightly-garbled plaintext-without-spaces, and its
# bazeries behaves as a small-alphabet substitution).
# ---------------------------------------------------------------------------
CIPHER_FAMILIES: dict[str, str] = {
    # Readable (or nearly readable) English
    "plaintext": "plain", "null_cipher": "plain",
    # One fixed letter-for-letter mapping: English "shape" survives
    "caesar": "mono_substitution", "caesar_rot": "mono_substitution",
    "rot13": "mono_substitution", "atbash": "mono_substitution",
    "affine": "mono_substitution", "monoalphabetic": "mono_substitution",
    "substitution": "mono_substitution", "kama_sutra": "mono_substitution",
    "bazeries": "mono_substitution", "joseon_yeokhak": "mono_substitution",
    "geez_monastic": "mono_substitution",
    # Repeating / progressive key over the plain alphabet
    "vigenere": "polyalphabetic", "beaufort": "polyalphabetic",
    "gronsfeld": "polyalphabetic", "porta": "polyalphabetic",
    "autokey": "polyalphabetic", "cardano_autokey": "polyalphabetic",
    "running_key": "polyalphabetic", "trithemius": "polyalphabetic",
    "confederate_vigenere": "polyalphabetic", "kryptos": "polyalphabetic",
    "alberti_disk": "polyalphabetic", "slidex": "polyalphabetic",
    "wheatstone": "polyalphabetic",
    # Letters unchanged, order scrambled
    "rail_fence": "transposition", "columnar": "transposition",
    "columnar_transposition": "transposition",
    "double_transposition": "transposition", "scytale": "transposition",
    "stager_route": "transposition",
    # Digraph / fractionating systems
    "playfair": "polygraphic", "four_square": "polygraphic",
    "two_square": "polygraphic", "hill": "polygraphic",
    "bifid": "polygraphic", "trifid": "polygraphic",
    "adfgx": "polygraphic", "adfgvx": "polygraphic",
    "fractionated_morse": "polygraphic",
    # Rotor machines, stream devices, and one-time systems: near-uniform output
    "enigma": "machine_or_otp", "typex": "machine_or_otp",
    "sigaba": "machine_or_otp", "kl7": "machine_or_otp",
    "fialka": "machine_or_otp", "purple": "machine_or_otp",
    "red_type_a": "machine_or_otp", "kryha": "machine_or_otp",
    "geheimschreiber": "machine_or_otp", "lorenz": "machine_or_otp",
    "m209": "machine_or_otp", "m94": "machine_or_otp",
    "jefferson_disk": "machine_or_otp", "diana": "machine_or_otp",
    "one_time_pad": "machine_or_otp", "vernam": "machine_or_otp",
    "solitaire": "machine_or_otp", "chaocipher": "machine_or_otp",
    # Distinct formats: digits, symbols, codebooks, non-Latin renders
    "morse_code": "code_format", "tap_code": "code_format",
    "pigpen": "code_format", "polybius": "code_format",
    "bacon_cipher": "code_format", "navajo_code": "code_format",
    "chinese_telegraph": "code_format", "commercial_code": "code_format",
    "jn25": "code_format", "zimmermann": "code_format",
    "culper_ring": "code_format", "book_cipher": "code_format",
    "arnold_andre": "code_format", "babington": "code_format",
    "aeneas_tacticus": "code_format", "wallis_cipher": "code_format",
    "nomenclator": "code_format", "homophonic": "code_format",
    "nihilist": "code_format", "straddling_checkerboard": "code_format",
    "vic": "code_format", "argenti": "code_format",
    "copiale": "code_format", "great_cipher": "code_format",
    "venona_pad_reuse": "code_format", "voynich_render": "code_format",
}


def label_family(label: str) -> str:
    """Return the statistical family for a cipher label ("unknown" if unmapped)."""
    return CIPHER_FAMILIES.get(label, "unknown")


# Uniform prior score so every known label appears in the output dict.
_PRIOR = 1.0 / len(_ALL_LABELS)


def _build_scores(**overrides: float) -> dict[str, float]:
    """Return a score dict seeded with the uniform prior and the given boosts."""
    s = {lbl: _PRIOR for lbl in _ALL_LABELS}
    for k, v in overrides.items():
        if k in s:
            s[k] = max(s[k], v)
    return s


def _norm_pred(scores: dict[str, float], source: str = "heuristic") -> ModelPrediction:
    total = sum(scores.values()) or 1.0
    norm = {k: round(v / total, 6) for k, v in scores.items()}
    label = max(norm, key=norm.get)
    return ModelPrediction(label, norm[label], norm, source)


def _deterministic(label: str, confidence: float) -> ModelPrediction:
    """Return a near-certain prediction with one dominant label."""
    fill = (1.0 - confidence) / max(len(_ALL_LABELS) - 1, 1)
    scores = {lbl: fill for lbl in _ALL_LABELS}
    scores[label] = confidence
    return ModelPrediction(label, confidence, scores, "heuristic")


def _load_calibration_map() -> list[tuple[float, float]]:
    """Load the isotonic confidence->accuracy breakpoints, or [] if absent."""
    path = Path(__file__).resolve().parent / "calibration_map.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    pts = sorted((float(k), float(v)) for k, v in
                 data.get("fine_confidence_to_accuracy", {}).items())
    return pts


_CALIBRATION_MAP = _load_calibration_map()


def calibrate_confidence(raw: float) -> float:
    """Map a raw heuristic confidence to its empirical accuracy via piecewise-
    linear interpolation over the isotonic calibration curve (from
    ``scripts/calibrate_confidence.py``). Identity if no map is present.

    Without this the raw confidences are badly miscalibrated — a raw 0.26
    corresponds to ~4% real accuracy, a raw 0.86 to ~97% — so an unvarnished
    number would mislead the very people the exhibit is teaching to weigh
    evidence.
    """
    pts = _CALIBRATION_MAP
    if not pts:
        return raw
    if raw <= pts[0][0]:
        return pts[0][1]
    if raw >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if x0 <= raw <= x1:
            t = (raw - x0) / (x1 - x0) if x1 > x0 else 0.0
            return round(y0 + t * (y1 - y0), 4)
    return raw


def heuristic_classify(text: str) -> ModelPrediction:
    """Transparent heuristic classifier with calibrated confidence.

    Thin wrapper over :func:`_heuristic_classify_raw` that replaces the raw
    (hand-tuned) confidence with its empirically-calibrated value, so a
    reported N% means "right about N% of the time" on the test distribution.
    """
    pred = _heuristic_classify_raw(text)
    cal = calibrate_confidence(pred.confidence)
    if cal == pred.confidence:
        return pred
    return ModelPrediction(pred.label, cal, pred.scores, pred.source)


def _heuristic_classify_raw(text: str) -> ModelPrediction:  # noqa: C901 – intentionally long
    """Multi-tier transparent heuristic classifier covering all 81 cipher labels.

    Tier 1 – Definitive character-set / format rules: non-alphabetic or highly
              constrained patterns that uniquely identify a cipher.
    Tier 2 – Statistical alphabet analysis: IoC, chi-squared, transposition
              signal, Friedman/Kasiski, and brute-force shifts for the ~50
              ciphers that produce plain A–Z lettertext.
    """
    stripped = text.strip()
    no_space = re.sub(r"\s+", "", stripped)

    # -----------------------------------------------------------------------
    # TIER 1a: Non-alphabetic distinct patterns
    # -----------------------------------------------------------------------

    # Tap code: only dots and spaces (no dashes), e.g. "... .....   .... .."
    if re.match(r"^[\s.]+$", stripped) and "." in stripped and "-" not in stripped:
        return _deterministic("tap_code", 0.93)

    # Morse code: dots, dashes, slashes, spaces – at least one dash
    if re.match(r"^[.\-/ \t\n]+$", stripped) and "-" in stripped:
        return _deterministic("morse_code", 0.93)

    # Pigpen cipher: contains distinctive geometric/box-drawing symbols
    _PIGPEN = set("¬•⌐┌┐├┬┴┼▽")
    if any(c in _PIGPEN for c in stripped):
        return _deterministic("pigpen", 0.91)

    # Babington: uses ⟨…⟩ token notation
    if "⟨" in stripped or "⟩" in stripped:
        return _deterministic("babington", 0.93)

    # Trifid: alphabetic but contains '+' as separator
    letters = clean_letters(text)
    if "+" in stripped and letters:
        return _deterministic("trifid", 0.88)

    # Navajo code: alpha text with '-' field separators (not morse – no dots)
    if "-" in stripped and "/" in stripped and letters and "." not in stripped:
        if len(set(letters)) > 8:
            return _deterministic("navajo_code", 0.83)

    # Venona pad reuse: letters + a strict digit subset {2, 4, 5}
    non_letter_non_space = set(c for c in no_space if not c.isalpha())
    # Venona pad reuse: letters mixed with digits drawn exclusively from {2–7}.
    # These specific digits correspond to the indicator groups used in the
    # Soviet one-time pad system.  93 % TP rate; < 1 % FP on other cipher types.
    if non_letter_non_space and non_letter_non_space.issubset({"2", "3", "4", "5", "6", "7"}) and letters:
        return _deterministic("venona_pad_reuse", 0.87)

    # Voynich manuscript: predominantly lowercase, short pseudo-syllabic tokens,
    # no uppercase (most other ciphers are uppercase).  Check before letter extraction.
    _voynich_words = {"oror", "oxed", "otol", "chedy", "shedy", "dainy", "otaiin",
                      "cheol", "raiin", "aiiin", "chol", "chor", "daral", "shal",
                      "daiin", "chedal", "sheedy", "okedy", "otain", "qokedy",
                      "okain", "shaiin", "sharal", "okal", "okshy"}
    # Broad English word set — if ANY of these appear the text is likely English
    # (null cipher) rather than alien Voynich script.
    _basic_english = {
        "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
        "are", "was", "its", "his", "her", "our", "can", "has", "had", "all",
        "one", "two", "new", "old", "now", "say", "who", "may", "use", "into",
        "from", "they", "what", "when", "each", "will", "also", "then", "come",
        "look", "call", "give", "over", "just", "know", "time", "long", "down",
        "most", "some", "take", "only", "good", "even", "back", "year", "much",
        "right", "name", "after", "little", "place", "great", "being", "same",
        "still", "found", "many", "should", "before", "other", "where", "while",
        "about", "people", "number", "sound", "sentence",
    }
    if stripped and stripped[0].islower():   # quick gate: ciphertext is lowercase
        lc_words = set(re.findall(r"[a-z]+", stripped.lower()))
        if len(lc_words) >= 3 and lc_words & _voynich_words:
            return _deterministic("voynich_render", 0.85)
        # Even without exact word matches, a fully-lowercase alpha-space text
        # that isn't English is likely voynich render.
        # Guard: if ANY basic English words appear, this is null_cipher, not voynich.
        if (all(c.islower() or c == " " for c in stripped)
                and len(stripped) > 8
                and word_score(stripped) < 2
                and not (lc_words & _basic_english)):
            return _deterministic("voynich_render", 0.72)

    # -----------------------------------------------------------------------
    # TIER 1b: Numeric / coded formats
    # -----------------------------------------------------------------------

    # Arnold-André / book_cipher triple format: "page.line.word"
    # Distinguish by line number: Arnold-André uses a small book (~6 lines/page),
    # book_cipher uses larger books (up to 50 lines/page).
    if re.match(r"^(\d+\.\d+\.\d+\s*)+$", stripped):
        triples = re.findall(r"(\d+)\.(\d+)\.(\d+)", stripped)
        if triples:
            max_line = max(int(t[1]) for t in triples)
            if max_line <= 8:
                return _deterministic("arnold_andre", 0.88)
            return _deterministic("book_cipher", 0.85)
        return _deterministic("arnold_andre", 0.89)

    # Book cipher: mixed "page.word" two-part codes, optionally mixed with bare integers
    if re.search(r"\d+\.\d+", stripped) and re.search(r"\d", stripped):
        tokens = stripped.split()
        if all(re.match(r"^\d+[.\d]*$", t) for t in tokens if t):
            return _deterministic("book_cipher", 0.85)

    # Pure numeric text (spaces allowed, no letters)
    digits_only = re.sub(r"\s", "", stripped)
    if digits_only and digits_only.isdigit() and not letters:
        tokens = [t for t in stripped.split() if t]
        if not tokens:
            pass  # fall through
        else:
            tok_lens = [len(t) for t in tokens if t.isdigit()]
            avg_len = sum(tok_lens) / len(tok_lens) if tok_lens else 0
            nums = [int(t) for t in tokens if t.isdigit()]

            # Polybius square: all tokens 2 digits from {1-5}
            if all(len(t) == 2 and t.isdigit() and all(c in "12345" for c in t)
                   for t in tokens):
                return _deterministic("polybius", 0.91)

            # Chinese telegraph: all tokens 4 digits
            if all(len(t) == 4 and t.isdigit() for t in tokens) and len(tokens) >= 3:
                return _deterministic("chinese_telegraph", 0.89)

            # JN-25 / naval code: all tokens 5 digits, values typically < 50 000
            # (codebook covers 00000-49999).  Zimmermann uses high 9XXXX groups.
            if all(len(t) == 5 and t.isdigit() for t in tokens):
                vals = [int(t) for t in tokens]
                above_50k = sum(1 for v in vals if v >= 50000)
                below_50k = sum(1 for v in vals if v < 50000)
                above_90k = sum(1 for v in vals if v >= 90000)
                # Zimmermann: heavy concentration of 9XXXX groups (90 000+)
                if above_90k / len(vals) >= 0.60:
                    return _deterministic("zimmermann", 0.85)
                # JN-25: all (or nearly all) values fall in the 0-49 999 range
                if below_50k / len(vals) >= 0.80 and above_50k == 0:
                    return _deterministic("jn25", 0.83)

            # Zimmermann also occurs with mixed 4–5-digit groups where many values ≥ 90 000
            if nums and all(len(t) in (4, 5) and t.isdigit() for t in tokens):
                above_90k_any = sum(1 for n in nums if n >= 90000)
                if above_90k_any / len(nums) >= 0.50:
                    return _deterministic("zimmermann", 0.78)

            # Culper Ring: 3-digit tokens overwhelmingly in the 800–999 range
            # (999 used as word separator, other tokens are letter codes 800–998)
            if all(len(t) == 3 for t in tokens if t.isdigit()):
                in_range = sum(1 for n in nums if 800 <= n <= 999)
                if len(nums) > 0 and in_range / len(nums) >= 0.70:
                    return _deterministic("culper_ring", 0.85)

            # Straddling checkerboard / VIC: long run of unspaced digits
            # VIC tends to be longer and more uniform than straddling_checkerboard
            if " " not in stripped and len(stripped) >= 10:
                if len(stripped) >= 35:
                    return _deterministic("vic", 0.70)
                return _deterministic("straddling_checkerboard", 0.72)

            # 2-digit token ciphers — distinguish by value range
            if all(len(t) == 2 for t in tokens) and len(tokens) >= 4:
                low_26 = sum(1 for n in nums if 1 <= n <= 26)
                high_50 = sum(1 for n in nums if 50 <= n <= 99)
                # Nomenclator: bimodal — majority are letter codes (01-26) mixed
                # with a few word codes (50-99).
                if low_26 / len(nums) >= 0.50 and high_50 >= 1:
                    return _deterministic("nomenclator", 0.65)
                # Aeneas Tacticus: all numbers 01–26 (pure alphabet positions)
                if all(1 <= n <= 26 for n in nums):
                    return _deterministic("aeneas_tacticus", 0.72)
                # Wallis cipher: uses 90 and/or 91 as explicit group markers
                # (they appear multiple times as structural delimiters).
                # Wallis rate is ~30 %; random homophonic rate is ~2 %, so
                # use a 10 % threshold to eliminate false positives.
                _w90 = (nums.count(90) + nums.count(91)) / len(nums)
                if _w90 >= 0.10:
                    return _deterministic("wallis_cipher", 0.70)
                # Nihilist: Polybius row+col sums → minimum possible value is 22
                # (smallest polybius cell 11 + smallest key cell 11). So a nihilist
                # ciphertext has NO values below 22.
                below_22 = sum(1 for n in nums if n < 22)
                in_nihilist_range = sum(1 for n in nums if 22 <= n <= 99)
                if below_22 == 0 and in_nihilist_range / len(nums) >= 0.80:
                    return _deterministic("nihilist", 0.65)
                # Homophonic: broad 00-99 distribution often includes values < 22
                if below_22 / len(nums) >= 0.15:
                    return _deterministic("homophonic", 0.50)
                return _deterministic("homophonic", 0.45)

            # Mixed / non-uniform token-length block
            # (reached when not all tokens are the same 2-digit length)
            if avg_len < 3.0 and len(tokens) >= 5:
                max_num = max(nums) if nums else 0
                # Wallis cipher markers still detectable in mixed-length text
                _w90 = (nums.count(90) + nums.count(91)) / len(nums)
                if _w90 >= 0.10:
                    return _deterministic("wallis_cipher", 0.62)
                if max_num <= 26:
                    return _deterministic("aeneas_tacticus", 0.55)
                # Nihilist overflow: sums can reach 110 (55+55); values are ≥ 22
                if max_num <= 110:
                    below_22 = sum(1 for n in nums if n < 22)
                    if below_22 == 0:
                        return _deterministic("nihilist", 0.55)
                # Broad 2-digit range without structure → homophonic
                if max_num <= 99:
                    return _deterministic("homophonic", 0.50)
                return _deterministic("homophonic", 0.45)

            if avg_len < 3.5:
                # 2–3 digit tokens: great_cipher, argenti
                return _deterministic("great_cipher", 0.50)

            # Default numeric fallback
            return _deterministic("argenti", 0.45)

    # Copiale cipher: letter + single digit pairs "I3 M4 O1 K7"
    if re.match(r"^([A-Za-z]\d\s+)*[A-Za-z]\d\s*$", stripped):
        return _deterministic("copiale", 0.91)

    # -----------------------------------------------------------------------
    # TIER 1c: Alphabetic but restricted character set
    # -----------------------------------------------------------------------
    letter_set = set(letters)

    if letters and len(letters) >= 10:
        # ADFGVX: only letters from {A, D, F, G, V, X}
        if letter_set <= {"A", "D", "F", "G", "V", "X"}:
            if "V" in letter_set:
                return _deterministic("adfgvx", 0.93)
            return _deterministic("adfgx", 0.91)  # {A,D,F,G,X} subset without V

        # ADFGX: only letters from {A, D, F, G, X}
        if letter_set <= {"A", "D", "F", "G", "X"}:
            return _deterministic("adfgx", 0.91)

        # Bacon cipher: only A and B (groups of 5)
        if letter_set <= {"A", "B"} and len(letters) >= 25:
            return _deterministic("bacon_cipher", 0.93)

    # Commercial code: ≥4 five-letter groups, many ending in Z/X
    word_groups = [w for w in text.upper().split() if re.match(r"^[A-Z]{5}$", w)]
    if len(word_groups) >= 4 and len(word_groups) / max(len(text.split()), 1) >= 0.55:
        zx_end = sum(1 for w in word_groups if w[-1] in "ZX")
        if zx_end / len(word_groups) >= 0.25:
            return _deterministic("commercial_code", 0.82)

    # Military 4-letter groups: vernam (XOR/OTP) or running_key (book cipher).
    # Require ALL alpha words to be exactly 4 letters and IoC < 0.048
    # (playfair / polygraphic ciphers also use 4-letter groups but have IoC >0.046).
    _al_words_raw = [w for w in stripped.upper().split() if re.match(r'^[A-Z]+$', w)]
    if len(_al_words_raw) >= 6 and len(letters) >= 24:
        _w4_count = sum(1 for w in _al_words_raw if len(w) == 4)
        _w4_ratio = _w4_count / len(_al_words_raw)
        if _w4_ratio >= 0.95:   # nearly ALL groups must be exactly 4 letters
            _cnt_early = Counter(letters)
            _n_early = len(letters)
            _ioc_early = (
                sum(v * (v - 1) for v in _cnt_early.values()) / max(1, _n_early * (_n_early - 1))
            )
            if _ioc_early < 0.048:   # exclude polygraphic range
                if _ioc_early < 0.040:
                    return _deterministic("vernam", 0.52)
                return _deterministic("running_key", 0.50)

    # -----------------------------------------------------------------------
    # TIER 2: Statistical analysis for pure alphabetic text
    # -----------------------------------------------------------------------
    # Always compute enough for brute-force checks (work on short text too).
    n_letters = len(letters)

    ev = analyze_evidence(text)
    ioc = ev.index_of_coincidence
    best_shift, best_chi, best_words, _ = ev.caesar_candidates[0]
    atbash_words = word_score(ev.atbash_plaintext)
    atbash_chi = chi_squared_for_english(clean_letters(ev.atbash_plaintext))
    affine_words = ev.affine_candidates[0][3] if ev.affine_candidates else 0
    raw_words = word_score(text)
    raw_chi = chi_squared_for_english(letters)
    fried = ev.friedman_key_length
    kas_support = sum(n for _, n in ev.kasiski_key_lengths[:3])
    transp = ev.transposition_signal
    bgm = ev.bigram_support
    _uniq = len(set(letters))  # unique A-Z letters for machine-cipher disambiguation

    # -----------------------------------------------------------------------
    # TIER 2a: Brute-force decodable checks (work even on short text)
    # -----------------------------------------------------------------------

    # Null cipher / plaintext: check FIRST so readable English is never
    # mis-classified.  A null cipher embeds the secret in specific positions
    # of an apparently-normal cover text, so "looks like English" is the
    # primary signature of both null_cipher and genuine plaintext.
    # Spaces required: run-together near-English is lorenz, not a cover text.
    if raw_words >= 3 and raw_chi < 150 and " " in stripped:
        return _deterministic("null_cipher", min(0.75, 0.15 + 0.06 * raw_words))

    # For short texts the word list may not produce 2 hits even for correct
    # decodes (e.g. "LIBERTY OR DEATH" → only "DEATH" matches).  Use a lower
    # threshold of 1 when the text is short enough that it would otherwise
    # fall through to too_short anyway.
    _short = n_letters < 25
    _atbash_thr = 1 if _short else 2
    _bf_thr = 1 if _short else 2

    # Atbash: word match OR chi-squared of decoded text is very low.
    # chi(atbash(atbash_text)) = chi(plaintext) ≈ 15-50 for any natural language.
    # chi for monoalphabetic/other: p10 ≥ 391 — threshold of 100 is safe.
    if atbash_words >= _atbash_thr or (n_letters >= 25 and atbash_chi < 100):
        return _deterministic("atbash", min(0.82, 0.25 + 0.07 * max(atbash_words, 1)))

    # ROT-13: shift-13 is a special case of Caesar
    # For non-English phrases, best_words may be 0 but best_chi is still very low.
    if best_shift == 13 and (best_words >= _bf_thr or (n_letters >= 25 and best_chi < 100)):
        return _deterministic("rot13", min(0.85, 0.25 + 0.08 * max(best_words, 1)))

    # Caesar / ROT-N: any non-zero shift that produces English
    if best_words >= _bf_thr and best_shift != 0:
        conf = min(0.82, 0.22 + 0.08 * best_words)
        if affine_words >= best_words + 2:
            return _deterministic("affine", min(0.78, 0.20 + 0.07 * affine_words))
        return _deterministic("caesar", conf)

    # Affine (even if Caesar has 0-1 words)
    affine_chi = ev.affine_candidates[0][2] if ev.affine_candidates else 999
    best_a = ev.affine_candidates[0][0] if ev.affine_candidates else 1
    if affine_words >= 3:
        return _deterministic("affine", min(0.75, 0.18 + 0.07 * affine_words))
    # Chi-based affine: best non-Caesar affine key decodes to natural language.
    # At threshold 50: affine TP=92%, monoalphabetic FP=1.7%, playfair FP=0%.
    # Exclude a=1 (caesar/rot13 already handled above).
    if n_letters >= 25 and affine_chi < 50 and best_a != 1:
        return _deterministic("affine", 0.65)

    # Short-text gate already handled above (n_letters < 12 → too_short).
    # For 12–19 letter texts we still fall through to IoC routing.

    # -----------------------------------------------------------------------
    # TIER 2b: IoC-based family routing (requires ≥ 12 letters)
    # -----------------------------------------------------------------------
    # Note: lowered threshold from 20 → 12 so that short polygraphic examples
    # (Playfair, Bifid, columnar) get IoC-family routing instead of too_short.
    if n_letters < 12:
        return ModelPrediction("too_short", 0.20, {"too_short": 0.20}, "heuristic")

    # --- Scytale / stager_route: X-padded transposition --------------------
    # These ciphers pad to fill a rectangular grid using null 'X' characters.
    # X-inflation raises chi, breaking the low-chi transposition check below.
    # Removing X's and testing chi of remaining letters reveals English dist.
    x_freq = letters.count("X") / max(1, n_letters)
    no_x_letters = letters.replace("X", "")
    x_chi = chi_squared_for_english(no_x_letters) if len(no_x_letters) >= 15 else 999
    if x_freq > 0.05 and x_chi < 80 and ioc >= 0.045:
        # Very high X-frequency with English non-X letters → padded transposition.
        return _deterministic("scytale", 0.48)

    # --- Transposition: English IoC + disrupted bigrams ---------------------
    # KEY INSIGHT: transposition ciphers keep English letter frequencies
    # (raw_chi LOW) while monoalphabetic raises chi (letters shift away from
    # English). Lorenz is near-plaintext with bgm ≈ 1.0 — excluded by bgm guard.
    _transp_by_chi = (ioc >= 0.055 and raw_chi < 80 and 0.20 <= bgm <= 0.70)
    _transp_by_signal = (transp >= 0.50 and bgm <= 0.40 and ioc >= 0.055)
    if _transp_by_chi or _transp_by_signal:
        # Transposition ciphers preserve letter frequencies → high IoC,
        # but scramble bigrams → lower bigram_support.
        # Use rail-fence scoring to distinguish rail_fence from columnar/double.
        # NOTE: all transpositions have English-like chi (letters are unchanged),
        # so chi alone can't discriminate. Use word_score on decoded candidates —
        # the correct rail count produces recognisable English words; columnar/double
        # with wrong rail counts won't.
        _rf_cands = best_rail_fence_candidates(letters, max_rails=10)
        if _rf_cands:
            _best_rf_words = max(c[3] for c in _rf_cands)
            if _best_rf_words >= 2:
                return _deterministic("rail_fence", 0.52)
        # Not rail_fence — route by length for columnar vs double transposition.
        # NOTE: a previous "IoC ≥ 0.063 → stager_route" gate was wrong ~30x more
        # often than right (all transpositions preserve English IoC; the exact
        # value is sampling noise).  stager_route is only claimed via the
        # X-padding rule above; otherwise the larger transposition classes win.
        if n_letters <= 150:
            return _deterministic("columnar_transposition", 0.45)
        else:
            return _deterministic("double_transposition", 0.40)

    # --- High IoC (≥ 0.058): monoalphabetic substitution family ------------
    if ioc >= 0.058:
        # Lorenz: near-plaintext cipher rendered as one run-together string with
        # occasional garbled chars.  Signature: very low chi (English letter
        # frequencies intact), very high bigram support, and NO spaces — spaced
        # readable English is null_cipher and was caught above.
        if raw_chi < 50 and bgm > 0.85 and " " not in stripped:
            return _deterministic("lorenz", 0.48)
        # NOTE: kama_sutra (paired-alphabet mono) is statistically identical to
        # any other monoalphabetic substitution, so there is no honest rule for
        # it; it falls through to the monoalphabetic branch below.  A previous
        # "spaced text, no word hits" rule was wrong 5x more often than right.
        if 0.058 <= ioc < 0.068:
            # Could be monoalphabetic substitution OR transposition with no bigram signal
            if transp >= 0.35 and bgm <= 0.50:
                return _deterministic("columnar_transposition", 0.38)
            # Joseon yeokhak: high chi (≈ 418), no Kasiski, limited alphabet ≤ 20.
            # Lower n_letters threshold vs older rule to catch shorter examples.
            if _uniq <= 20 and n_letters >= 25 and raw_chi > 200 and kas_support == 0:
                return _deterministic("joseon_yeokhak", 0.30)
            # Wheatstone: rotor-based monoalphabetic, IoC near 0.060-0.065.
            # Corpus wheatstone is one run-together block; spaced text in this
            # regime is far more likely a plain monoalphabetic substitution.
            if 0.058 <= ioc < 0.065 and _uniq <= 22 and n_letters >= 30 and " " not in stripped:
                return _deterministic("wheatstone", 0.30)
            return _deterministic("monoalphabetic", 0.38)
        # IoC ≥ 0.068 — very high; scytale/stager_route have high IoC
        # Geez monastic: very high chi (≈ 501), robust Kasiski signal (≥ 4), limited alphabet.
        if raw_chi > 300 and kas_support >= 4 and _uniq <= 18:
            return _deterministic("geez_monastic", 0.32)
        if ioc >= 0.070:
            if transp >= 0.40:
                return _deterministic("scytale", 0.38)
        return _deterministic("monoalphabetic", 0.36)

    # --- Medium IoC (0.046–0.058): polygraphic / Playfair / polyalphabetic ---
    if 0.046 <= ioc < 0.058:
        # Bazeries: polyalphabetic with very limited active alphabet.
        # Short keys over a Napoleon-square alphabet reduce unique letters to ≤12.
        if _uniq <= 12 and n_letters >= 30:
            return _deterministic("bazeries", 0.38)
        # Cardano autokey: IoC near lower end (≈ 0.047), high chi (≈ 376), Kasiski support.
        # Distinguish from standard porta/playfair by lower IoC + high chi.
        if ioc < 0.050 and raw_chi > 250 and kas_support >= 2:
            return _deterministic("cardano_autokey", 0.28)
        # Porta cipher: medium-high IoC (0.049-0.058), very high chi (≈ 481).
        # Reciprocal alphabet pairs give strong Kasiski signal and non-English chi.
        # Removed old fried constraint (fried_med ≈ 1.4 — below old 2 <= fried <= 13).
        if 0.049 <= ioc < 0.058 and raw_chi > 200 and kas_support >= 2:
            return _deterministic("porta", 0.32)
        # Fractionated Morse: high-end medium IoC (≈ 0.057), elevated chi (≈ 295),
        # Kasiski support ≥ 3 (trigraph period from Morse encoding), limited uniq.
        if ioc >= 0.053 and raw_chi > 180 and kas_support >= 3 and _uniq <= 22:
            return _deterministic("fractionated_morse", 0.30)
        if 0.049 <= ioc < 0.058:
            return _deterministic("playfair", 0.32)
        return _deterministic("playfair", 0.35)

    # --- Low-to-medium IoC (0.040–0.046): polyalphabetic & machine ---------
    if 0.040 <= ioc < 0.046:
        # High chi in this IoC range signals non-English frequency distribution —
        # rules out standard Vigenere of English plaintext (chi ≈ 20-80).
        # Length gate: below ~60 letters chi/IoC are too noisy to justify an
        # exotic machine-class verdict — short texts fall through to the large
        # polyalphabetic classes instead.
        if raw_chi > 130 and n_letters >= 60:
            # Slidex: strong Kasiski support + very high chi (≈ 554)
            if kas_support >= 3 and raw_chi > 350:
                return _deterministic("slidex", 0.28)
            if kas_support == 0:
                # Jefferson disk: maximum chi in this bucket (≈ 727), no Kasiski
                if raw_chi > 450:
                    return _deterministic("jefferson_disk", 0.28)
                # Confederate Vigenere: high chi (≈ 409), higher-end IoC (≈ 0.044), no Kasiski
                if raw_chi > 280 and ioc >= 0.043:
                    return _deterministic("confederate_vigenere", 0.28)
                # Hill cipher: high chi (≈ 393), lower-end IoC (≈ 0.042), no Kasiski
                if raw_chi > 250 and ioc < 0.043:
                    return _deterministic("hill", 0.28)
                # Bifid: moderate chi (≈ 190), low-end IoC (≈ 0.042), no Kasiski
                if raw_chi > 130 and ioc < 0.043:
                    return _deterministic("bifid", 0.28)
        if 2 <= fried <= 12:
            # Gronsfeld uses digit key (0-9), giving short key lengths (1-6 typical).
            # Its IoC falls here (same as Vigenère), but key is shorter than most
            # Vigenère keys, making Friedman estimate cluster near 2-6.
            if 2 <= fried <= 6 and kas_support >= 2:
                return _deterministic("gronsfeld", 0.38)
            if kas_support >= 3:
                return _deterministic("vigenere", 0.42)
            # Weak Kasiski + Friedman → machine cipher or weak Vigenere.
            # Machine ciphers (purple, red_type_a, fialka, geheimschreiber, diana)
            # tend to have no strong Kasiski pattern (kas_support 0-1).
            if kas_support <= 1 and n_letters >= 60:
                if ioc < 0.042:
                    return _deterministic("purple", 0.28)
                if 0.042 <= ioc < 0.044:
                    return _deterministic("geheimschreiber", 0.28)
                return _deterministic("diana", 0.28)
            return _deterministic("vigenere", 0.36)
        # Beaufort: reciprocal Vigenère, same IoC range but Kasiski less regular.
        # When Friedman estimate is out of normal polyalphabetic range, lean Beaufort.
        if kas_support == 0:
            # Machine cipher with near-random output and no Kasiski
            if ioc < 0.042 and n_letters >= 60:
                if _uniq <= 18:
                    return _deterministic("red_type_a", 0.28)
                return _deterministic("fialka", 0.28)
            return _deterministic("beaufort", 0.32)
        return _deterministic("beaufort", 0.28)

    # --- Very low IoC (< 0.040): machine ciphers / OTP / autokey / stream --
    if ioc < 0.040:
        # Trithemius: progressive Caesar (shift = position index).  Decrypting
        # with the inverse progressive key produces English with low chi and
        # recognisable words.  This check fires before the enigma catch-all.
        if n_letters >= 25:
            _tri_dec = "".join(
                chr((ord(c) - 65 - i % 26) % 26 + 65)
                for i, c in enumerate(letters)
            )
            if chi_squared_for_english(_tri_dec) < 50 and word_score(_tri_dec) >= 3:
                return _deterministic("trithemius", 0.60)
        # Autokey (running key): the key = keyword + plaintext, so key length ≈
        # message length.  Friedman over-estimates key length (→ very large fried).
        # IoC is slightly above pure random (0.036–0.042) because key contains
        # English letter frequencies.  Kasiski finds almost no repeats (kas_support ≈ 0).
        if ioc >= 0.036 and kas_support == 0 and fried >= max(15, n_letters * 0.4):
            return _deterministic("autokey", 0.38)
        # Split the near-random machine-cipher space by IoC sub-range.
        # All these ciphers produce flat letter distributions; we use IoC and
        # unique-letter count as rough discriminators. This gives non-zero recall
        # on each class vs. the previous enigma catch-all.
        if ioc < 0.026:
            return _deterministic("chaocipher", 0.28)  # Chaocipher: extremely flat, IoC ≈ 0.020
        if ioc < 0.031:
            return _deterministic("m209", 0.27)        # M-209 rotor: IoC ≈ 0.028
        # One-time pad: maximum chi (≈ 686) among all ciphers, all 25-26 letters
        # represented (uniq ≥ 24), no Kasiski repeats. Most entropic output.
        if raw_chi > 550 and _uniq >= 24 and kas_support == 0:
            return _deterministic("one_time_pad", 0.30)
        return _deterministic("enigma", 0.25)

    # --- Catch-all for anything remaining ----------------------------------
    return _deterministic("vigenere", 0.22)



def build_explanation(text: str, pred: ModelPrediction) -> str:
    ev = analyze_evidence(text)
    lines = [
        f"### Detective conclusion: `{pred.label}`",
        f"**Cipher family:** `{label_family(pred.label)}` — family verdicts are far more "
        "reliable than exact labels; several ciphers are statistically identical.  ",
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

