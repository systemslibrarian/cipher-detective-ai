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
    # Top-10 English bigrams (TH, HE, IN, …) are ~12–15% of all bigrams in real
    # text. The previous 0.06 constant saturated the ratio on any vaguely-English
    # input — including most columnar transpositions — and zeroed out the signal.
    bigram_support = min(1.0, common_hits / (total_bigrams * 0.12))
    transposition = english_likeness * (1.0 - bigram_support)
    return round(transposition, 4), round(bigram_support, 4)


# ---------------------------------------------------------------------------
# Hill-climbing solver for monoalphabetic substitution
# ---------------------------------------------------------------------------

# Approximate English bigram log-probabilities. These are coarse — drawn from a
# small training corpus — but enough to give a hill-climbing solver a useful
# gradient. Unseen bigrams fall back to summed unigram log-probs (a standard
# backoff trick) so the gradient stays smooth instead of plateauing on a floor.
_BIGRAM_LOG_PROB: Dict[str, float] = {
    "TH": -2.31, "HE": -2.43, "IN": -2.78, "ER": -2.90, "AN": -3.00, "RE": -3.05,
    "ON": -3.18, "AT": -3.25, "EN": -3.27, "ND": -3.30, "TI": -3.40, "ES": -3.45,
    "OR": -3.48, "TE": -3.55, "OF": -3.60, "ED": -3.65, "IS": -3.70, "IT": -3.72,
    "AL": -3.75, "AR": -3.78, "ST": -3.80, "TO": -3.82, "NT": -3.85, "NG": -3.90,
    "SE": -3.95, "HA": -3.98, "AS": -4.00, "OU": -4.05, "IO": -4.08, "LE": -4.10,
    "VE": -4.15, "CO": -4.18, "ME": -4.20, "DE": -4.25, "HI": -4.28, "RI": -4.30,
    "RO": -4.32, "IC": -4.35, "NE": -4.38, "EA": -4.40, "RA": -4.42, "CE": -4.45,
    "LI": -4.48, "CH": -4.50, "LL": -4.55, "BE": -4.58, "MA": -4.60, "SI": -4.62,
    "OM": -4.65, "UR": -4.68,
}
# log10 of English letter frequency / 100 — used as a backoff for unseen bigrams.
_UNIGRAM_LOG_PROB: Dict[str, float] = {
    ch: math.log10(max(freq, 0.01) / 100.0) for ch, freq in ENGLISH_FREQ.items()
}
_BIGRAM_FLOOR = -8.0
_BIGRAM_BACKOFF_PENALTY = 1.0  # subtracted from unigram fallback so seen pairs win


def english_bigram_score(letters: str) -> float:
    """Mean log-probability per bigram. Higher (less negative) = more English-like.

    Uses a small bigram table for common pairs; falls back to (log p(a) + log p(b))
    minus a small penalty for unseen bigrams. The backoff keeps the gradient
    smooth so hill-climbing doesn't plateau on a floor.
    """
    if len(letters) < 2:
        return _BIGRAM_FLOOR
    total = 0.0
    pairs = len(letters) - 1
    for i in range(pairs):
        bg = letters[i:i + 2]
        seen = _BIGRAM_LOG_PROB.get(bg)
        if seen is not None:
            total += seen
        else:
            total += (
                _UNIGRAM_LOG_PROB.get(bg[0], -3.0)
                + _UNIGRAM_LOG_PROB.get(bg[1], -3.0)
                - _BIGRAM_BACKOFF_PENALTY
            )
    return total / pairs


def _apply_key(letters: str, key: str) -> str:
    """Apply a 26-letter substitution key (cipher A..Z -> plaintext)."""
    return letters.translate(str.maketrans(ALPHABET, key))


def hill_climb_substitution(
    text: str,
    iterations: int = 4000,
    restarts: int = 3,
    seed: int | None = 42,
) -> Tuple[str, str, float]:
    """Solve monoalphabetic substitution by hill climbing on bigram log-prob.

    Returns ``(plaintext_guess, key, score)``. ``key`` is a 26-letter string
    where ``key[i]`` is the plaintext letter for cipher letter ``ALPHABET[i]``.

    Educational only — works on a few hundred letters of English, fails on
    short or non-English samples. That failure mode is part of the lesson.
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
    for cipher_letter, plain_letter in zip(observed, english_order):
        seed_key[ALPHABET.index(cipher_letter)] = plain_letter

    best_key = "".join(seed_key)
    best_score = english_bigram_score(_apply_key(letters, best_key))

    for restart in range(max(1, restarts)):
        current = list(seed_key) if restart == 0 else list(best_key)
        if restart > 0:
            # Randomise more aggressively on each restart so we escape local optima.
            for _ in range(2 + restart):
                a, b = rng.sample(range(26), 2)
                current[a], current[b] = current[b], current[a]
        current_score = english_bigram_score(_apply_key(letters, "".join(current)))
        no_improve = 0
        for _ in range(iterations):
            a, b = rng.sample(range(26), 2)
            current[a], current[b] = current[b], current[a]
            score = english_bigram_score(_apply_key(letters, "".join(current)))
            if score > current_score:
                current_score = score
                no_improve = 0
            else:
                current[a], current[b] = current[b], current[a]  # revert
                no_improve += 1
                if no_improve > iterations // 4:
                    break
        if current_score > best_score:
            best_score = current_score
            best_key = "".join(current)

    plaintext = _apply_key(text.upper(), best_key)
    return plaintext, best_key, best_score


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


# ---------------------------------------------------------------------------
# All 81 cipher labels present in the full dataset.
# ---------------------------------------------------------------------------
_ALL_LABELS: List[str] = [
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

# Uniform prior score so every known label appears in the output dict.
_PRIOR = 1.0 / len(_ALL_LABELS)


def _build_scores(**overrides: float) -> Dict[str, float]:
    """Return a score dict seeded with the uniform prior and the given boosts."""
    s = {lbl: _PRIOR for lbl in _ALL_LABELS}
    for k, v in overrides.items():
        if k in s:
            s[k] = max(s[k], v)
    return s


def _norm_pred(scores: Dict[str, float], source: str = "heuristic") -> ModelPrediction:
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


def heuristic_classify(text: str) -> ModelPrediction:  # noqa: C901 – intentionally long
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
    if non_letter_non_space and non_letter_non_space.issubset({"2", "4", "5"}) and letters:
        return _deterministic("venona_pad_reuse", 0.87)

    # -----------------------------------------------------------------------
    # TIER 1b: Numeric / coded formats
    # -----------------------------------------------------------------------

    # Arnold-André: "page.line.word" triples, e.g. "4.2.4 13.1.1"
    if re.match(r"^(\d+\.\d+\.\d+\s*)+$", stripped):
        return _deterministic("arnold_andre", 0.89)

    # Book cipher: mixed "page.word" or "page:line:word" with periods
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

            # Polybius square: all tokens 2 digits from {1-5}
            if all(len(t) == 2 and t.isdigit() and all(c in "12345" for c in t)
                   for t in tokens):
                return _deterministic("polybius", 0.91)

            # Chinese telegraph: all tokens 4 digits
            if all(len(t) == 4 and t.isdigit() for t in tokens) and len(tokens) >= 3:
                return _deterministic("chinese_telegraph", 0.89)

            # JN-25 / naval code: all tokens 5 digits, many start with 0
            if all(len(t) == 5 and t.isdigit() for t in tokens):
                leading0 = sum(1 for t in tokens if t.startswith("0"))
                if leading0 / len(tokens) >= 0.4:
                    return _deterministic("jn25", 0.83)
                # Zimmermann: 5-digit groups with many 9s or 0s
                heavy = sum(1 for t in tokens if t[0] in "90")
                if heavy / len(tokens) >= 0.5:
                    return _deterministic("zimmermann", 0.80)

            # Straddling checkerboard / VIC: long run of unspaced digits
            if " " not in stripped and len(stripped) >= 20:
                return _deterministic("straddling_checkerboard", 0.72)

            # VIC cipher: all-digit string with spaces, high density
            if len(tokens) >= 5 and avg_len < 2.5:
                # 1-2 digit tokens – aeneas_tacticus, nihilist, homophonic,
                # nomenclator, wallis_cipher; pick most common in dataset
                if all(len(t) == 2 for t in tokens):
                    return _deterministic("nihilist", 0.52)
                return _deterministic("aeneas_tacticus", 0.48)

            if avg_len < 3.5:
                # 2–3 digit tokens: culper_ring, great_cipher, argenti, wallis
                if avg_len < 3.0:
                    return _deterministic("culper_ring", 0.50)
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

    # -----------------------------------------------------------------------
    # TIER 2: Statistical analysis for pure alphabetic text
    # -----------------------------------------------------------------------
    if len(letters) < 20:
        return ModelPrediction("too_short", 0.20, {"too_short": 0.20}, "heuristic")

    ev = analyze_evidence(text)
    ioc = ev.index_of_coincidence
    best_shift, best_chi, best_words, _ = ev.caesar_candidates[0]
    atbash_words = word_score(ev.atbash_plaintext)
    affine_words = ev.affine_candidates[0][3] if ev.affine_candidates else 0
    raw_words = word_score(text)
    raw_chi = chi_squared_for_english(letters)
    fried = ev.friedman_key_length
    kas_support = sum(n for _, n in ev.kasiski_key_lengths[:3])
    n_letters = len(letters)

    # -----------------------------------------------------------------------
    # Build a fresh score dict and apply tiered statistical signals.
    # The strategy: detect the IoC-based cipher FAMILY first, then apply
    # brute-force / structural disambiguation within that family.
    # -----------------------------------------------------------------------

    # --- Plaintext: check FIRST so readable English is never mis-classified ---
    # Affine brute-force includes (a=1, b=0) which is the identity, so it will
    # always find English words in actual plaintext — plaintext must fire first.
    if raw_words >= 3 and raw_chi < 150:
        return _deterministic("plaintext", min(0.82, 0.18 + 0.08 * raw_words))

    # --- Brute-force decodable monoalphabetic signals (fast, high confidence) ---

    # Atbash: one-step decode
    if atbash_words >= 2:
        return _deterministic("atbash", min(0.82, 0.25 + 0.07 * atbash_words))

    # ROT-13: shift-13 is a special case of Caesar
    if best_shift == 13 and best_words >= 2:
        return _deterministic("rot13", min(0.85, 0.25 + 0.08 * best_words))

    # Caesar / ROT-N: any non-zero shift that produces English
    if best_words >= 2 and best_shift != 0:
        conf = min(0.82, 0.22 + 0.08 * best_words)
        # Affine: only promote if it's strictly better than Caesar
        if affine_words >= best_words + 2:
            return _deterministic("affine", min(0.78, 0.20 + 0.07 * affine_words))
        return _deterministic("caesar", conf)

    # Affine (even if caesar has 0-1 words, try affine if it finds English)
    if affine_words >= 3:
        return _deterministic("affine", min(0.75, 0.18 + 0.07 * affine_words))

    # -----------------------------------------------------------------------
    # At this point no simple decode worked. Use IoC-based family routing.
    # -----------------------------------------------------------------------
    transp = ev.transposition_signal
    bgm = ev.bigram_support

    # --- Transposition: English IoC + disrupted bigrams ---------------------
    if transp >= 0.50 and bgm <= 0.40 and ioc >= 0.055:
        # Transposition ciphers preserve letter frequencies → high IoC,
        # but scramble bigrams → low bigram_support.
        if n_letters <= 70:
            return _deterministic("rail_fence", 0.45)
        elif n_letters <= 180:
            return _deterministic("columnar_transposition", 0.40)
        else:
            return _deterministic("double_transposition", 0.35)

    # --- High IoC (≥ 0.058): monoalphabetic substitution family ------------
    if ioc >= 0.058:
        if 0.058 <= ioc < 0.068:
            # Could be monoalphabetic substitution OR transposition with no bigram signal
            if transp >= 0.35 and bgm <= 0.50:
                return _deterministic("columnar_transposition", 0.38)
            return _deterministic("monoalphabetic", 0.38)
        # IoC ≥ 0.068 — very high, unusual; scytale/stager_route have high IoC
        if ioc >= 0.070:
            if transp >= 0.40:
                return _deterministic("scytale", 0.38)
        return _deterministic("monoalphabetic", 0.36)

    # --- Medium IoC (0.046–0.058): polygraphic / Playfair family -----------
    if 0.046 <= ioc < 0.058:
        # Fractionated Morse: constrained letter set + medium IoC
        frac_morse_letters = {
            "A", "D", "F", "G", "H", "I", "L", "M", "N", "O", "R", "S", "T", "U", "W"
        }
        if letter_set <= frac_morse_letters:
            return _deterministic("fractionated_morse", 0.52)
        # Gronsfeld: uses digit-based repeating key → low IoC similar to Vigenère
        if 0.049 <= ioc < 0.058:
            return _deterministic("gronsfeld", 0.30) if (2 <= fried <= 8) else _deterministic("playfair", 0.32)
        return _deterministic("playfair", 0.35)

    # --- Low-to-medium IoC (0.040–0.046): polyalphabetic & some machine ----
    if 0.040 <= ioc < 0.046:
        # Friedman key-length estimate distinguishes repeating-key ciphers
        # from one-time / aperiodic stream ciphers.
        if 2 <= fried <= 12:
            if kas_support >= 3:
                return _deterministic("vigenere", 0.42)
            return _deterministic("vigenere", 0.36)
        return _deterministic("beaufort", 0.28)

    # --- Very low IoC (< 0.040): machine ciphers / OTP / strong stream -----
    # enigma is the most common label in this IoC band (1 887 examples).
    if ioc < 0.040:
        return _deterministic("enigma", 0.25)

    # --- Catch-all for anything remaining ----------------------------------
    return _deterministic("vigenere", 0.22)



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

