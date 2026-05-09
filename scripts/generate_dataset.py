from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
AFFINE_A_VALUES = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]

# ---------------------------------------------------------------------------
# Attack-method and educational-note registries (extended for all new labels)
# ---------------------------------------------------------------------------
ATTACKS: Dict[str, List[str]] = {
    "plaintext":             [],
    "caesar_rot":            ["brute_force_26", "frequency_analysis", "chi_squared_english"],
    "caesar":                ["brute_force_26", "frequency_analysis", "chi_squared_english"],
    "rot13":                 ["brute_force_26", "self_inverse_check"],
    "atbash":                ["self_inverse_check", "frequency_analysis"],
    "affine":                ["brute_force_312", "frequency_analysis"],
    "monoalphabetic":        ["frequency_analysis", "pattern_words", "hill_climbing_lm"],
    "substitution":          ["frequency_analysis", "pattern_words", "hill_climbing_lm"],
    "vigenere":              ["kasiski_examination", "friedman_test", "index_of_coincidence"],
    "beaufort":              ["kasiski_examination", "friedman_test", "index_of_coincidence"],
    "gronsfeld":             ["kasiski_examination", "digit_key_brute_force"],
    "autokey":               ["running_key_analysis", "index_of_coincidence"],
    "trithemius":            ["progressive_shift_detection", "frequency_analysis"],
    "porta":                 ["kasiski_examination", "index_of_coincidence"],
    "rail_fence":            ["rail_count_search", "anagram_recovery"],
    "columnar":              ["key_length_search", "anagram_recovery"],
    "columnar_transposition":["key_length_search", "anagram_recovery"],
    "scytale":               ["column_count_search", "anagram_recovery"],
    "double_transposition":  ["key_length_search", "anagram_recovery"],
    "stager_route":          ["column_count_search", "route_pattern_search"],
    # new sparse-label entries
    "aeneas_tacticus":       ["sequential_number_decode", "frequency_analysis"],
    "arnold_andre":          ["book_cipher_attack", "triple_index_search"],
    "babington":             ["nomenclator_frequency", "symbol_mapping"],
    "bacon_cipher":          ["five_bit_decode", "ab_frequency_analysis"],
    "book_cipher":           ["book_cipher_attack", "triple_index_search"],
    "commercial_code":       ["codebook_frequency", "five_letter_group_analysis"],
    "culper_ring":           ["codebook_frequency", "number_range_analysis"],
    "homophonic":            ["frequency_analysis", "multiple_ciphertext_attack"],
    "morse_code":            ["symbol_decode", "frequency_analysis"],
    "navajo_code":           ["codebook_frequency", "word_list_attack"],
    "null_cipher":           ["acrostic_detection", "first_letter_extraction"],
    "one_time_pad":          ["information_theoretic_proof", "key_reuse_attack"],
    "pigpen":                ["symbol_frequency", "grid_position_decode"],
    "polybius":              ["pair_decode", "frequency_analysis"],
    "running_key":           ["index_of_coincidence", "probable_plaintext"],
    "tap_code":              ["dot_count_decode", "grid_position_search"],
    "vernam":                ["key_reuse_attack", "information_theoretic_proof"],
    "voynich_render":        ["statistical_analysis", "unsolved"],
    "wallis_cipher":         ["number_substitution_decode", "frequency_analysis"],
    "zimmermann":            ["codebook_frequency", "number_group_analysis"],
}

EDU_NOTES: Dict[str, str] = {
    "plaintext":             "No cipher applied. Useful as a negative-class baseline.",
    "caesar_rot":            "Caesar / ROT-N: single-shift monoalphabetic with only 26 keys.",
    "caesar":                "Caesar / ROT-N: single-shift monoalphabetic with only 26 keys.",
    "rot13":                 "ROT-13 is Caesar with shift 13 — it is its own inverse.",
    "atbash":                "Atbash maps A↔Z, B↔Y, … and is its own inverse.",
    "affine":                "Affine: E(x)=(a·x+b) mod 26 — only 312 valid keys.",
    "monoalphabetic":        "Monoalphabetic substitution with an arbitrary 26-letter permutation.",
    "substitution":          "Monoalphabetic substitution with an arbitrary 26-letter permutation.",
    "vigenere":              "Vigenère: repeating-key Caesar; vulnerable to Kasiski and Friedman analysis.",
    "beaufort":              "Beaufort: E(x)=k-x mod 26 — reciprocal cipher (decryption identical to encryption).",
    "gronsfeld":             "Gronsfeld: Vigenère variant with a digit-only key (0–9 per position).",
    "autokey":               "Autokey: Vigenère where the plaintext extends the key — harder to Kasiski-attack.",
    "trithemius":            "Trithemius: progressive Caesar (shift = letter position); no key needed.",
    "porta":                 "Porta: 13-row table polyalphabetic cipher; reciprocal (encrypt = decrypt).",
    "rail_fence":            "Rail-fence: zig-zag transposition — rearranges letters, keeps frequencies.",
    "columnar":              "Columnar transposition: writes rows, reads columns in key order.",
    "columnar_transposition":"Columnar transposition: writes rows, reads columns in key order.",
    "scytale":               "Scytale: Spartan cylinder cipher — wrap text around a rod of fixed diameter.",
    "double_transposition":  "Double columnar transposition: apply columnar twice for stronger security.",
    "stager_route":          "Route (Stager) cipher: write into rectangle, read via alternating column route.",
    # new sparse-label entries
    "aeneas_tacticus":       "Aeneas Tacticus: Greek sequential-number cipher; A=1…Z=26.",
    "arnold_andre":          "Arnold-André book cipher: coordinates (page.section.word) referencing a shared text.",
    "babington":             "Babington nomenclator: symbol tokens ⟨wNN⟩/⟨aNN⟩ used in the 1586 Babington Plot.",
    "bacon_cipher":          "Bacon's biliteral cipher: each letter encoded as 5-bit A/B string.",
    "book_cipher":           "Book cipher: coordinates (page.line.word) referencing a shared document.",
    "commercial_code":       "Commercial code: five-letter groups from a codebook, often ending in X or Z.",
    "culper_ring":           "Culper Ring code: Washington's three-digit codebook numbers (800–999 range).",
    "homophonic":            "Homophonic substitution: each letter maps to multiple possible numbers.",
    "morse_code":            "Morse code: dots and dashes; word-separated by / symbol.",
    "navajo_code":           "Navajo Code Talker cipher: Navajo words represent English letters.",
    "null_cipher":           "Null/steganographic cipher: hidden message formed by first letters of words.",
    "one_time_pad":          "One-time pad: theoretically unbreakable if key is truly random and never reused.",
    "pigpen":                "Pigpen cipher: letters encoded as positions in tic-tac-toe and X grids.",
    "polybius":              "Polybius square: 5×5 grid maps letters to two-digit row-column pairs.",
    "running_key":           "Running-key cipher: Vigenère with a very long key (e.g. a book passage).",
    "tap_code":              "Tap code: letters encoded as dot-groups representing row and column in 5×5 grid.",
    "vernam":                "Vernam cipher: XOR-based stream cipher; with a random key equivalent to OTP.",
    "voynich_render":        "Voynich manuscript: undeciphered 15th-century script; pseudo-syllabic output.",
    "wallis_cipher":         "Wallis cipher: each letter mapped to a unique two-digit number from a fixed table.",
    "zimmermann":            "Zimmermann Telegram code: WWI German diplomatic numeric codebook cipher.",
}

# ---------------------------------------------------------------------------
# Diverse plaintext corpus (mix of historical, technical, and educational text)
# ---------------------------------------------------------------------------
BASE_SENTENCES = [
    "THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY",
    "CRYPTOGRAPHY REWARDS PATIENCE AND CAREFUL THINKING",
    "EVERY SYSTEM NEEDS AN HONEST THREAT MODEL",
    "CLASSICAL CIPHERS TEACH WHY MODERN SECURITY MATTERS",
    "THE DETECTIVE STUDIES PATTERNS BEFORE MAKING CLAIMS",
    "FAIL CLOSED IS A POSTURE NOT A SLOGAN",
    "FREQUENCY ANALYSIS CAN REVEAL WEAK CIPHERS",
    "THE MESSAGE WAS HIDDEN BUT NOT SECURE",
    "GOOD EDUCATIONAL TOOLS EXPLAIN THEIR LIMITS",
    "PUBLIC INFRASTRUCTURE DESERVES CAREFUL DESIGN",
    "A STRONG CLAIM NEEDS STRONG EVIDENCE",
    "THE INDEX OF COINCIDENCE IS A USEFUL CLUE",
    "HISTORICAL CIPHERS ARE TEACHERS NOT PROTECTION",
    "A TRANSPARENT TOOL SHOULD SHOW ITS REASONING",
    "SECURITY CLAIMS SHOULD BE TESTED UNDER SCRUTINY",
    "A MODEL CAN CLASSIFY PATTERNS BUT IT CAN ALSO BE WRONG",
    "MODERN ENCRYPTION IS NOT BROKEN BY LETTER FREQUENCY",
    "THE BEST LESSONS SHOW WHERE THE METHOD FAILS",
    "OBSERVABLE EVIDENCE COMES BEFORE CONFIDENCE",
    "THE ARCHITECTURE SHOULD MAKE MISUSE HARDER",
    "TRANSPOSITION CIPHERS REARRANGE LETTERS WITHOUT SUBSTITUTION",
    "THE KASISKI EXAMINATION FINDS REPEATED TRIGRAMS IN CIPHERTEXT",
    "POLYALPHABETIC CIPHERS USE MULTIPLE ALPHABET SHIFTS",
    "SUBSTITUTION CIPHERS MAP EACH LETTER TO A DIFFERENT LETTER",
    "THE RAIL FENCE CIPHER WRITES TEXT IN A ZIGZAG PATTERN",
    "COLUMNAR TRANSPOSITION READS COLUMNS IN KEY ORDER",
    "AN AFFINE CIPHER APPLIES A LINEAR FUNCTION TO EACH LETTER",
    "THE VIGENERE CIPHER USES A REPEATING KEYWORD",
    "CAESAR SHIFTED THE ALPHABET BY THREE POSITIONS",
    "ATBASH ENCODES BY REVERSING THE ALPHABET",
    "INDEX OF COINCIDENCE MEASURES LETTER FREQUENCY DISTRIBUTION",
    "FRIEDMAN ESTIMATED KEY LENGTH FROM STATISTICS",
    "BRUTE FORCE WORKS ONLY WHEN THE KEY SPACE IS SMALL",
    "PATTERN WORDS HELP BREAK MONOALPHABETIC SUBSTITUTION",
    "ENTROPY MEASURES UNCERTAINTY IN A DISTRIBUTION",
    "DIGRAPH AND TRIGRAPH FREQUENCIES EXPOSE SIMPLE CIPHERS",
    "HUMAN READABLE OUTPUT SHOULD SHOW ALL REASONING STEPS",
    "DO NOT CONFUSE OBFUSCATION WITH ENCRYPTION",
    "PEER REVIEW AND REPRODUCIBILITY MATTER IN RESEARCH",
    "OPEN SOURCE TOOLS ALLOW INDEPENDENT VERIFICATION",
    "THE ENIGMA MACHINE USED ROTORS TO CREATE A POLYALPHABETIC CIPHER",
    "ALAN TURING BROKE ENIGMA WITH CRIBS AND THE BOMBE MACHINE",
    "THE PLAYFAIR CIPHER ENCRYPTS PAIRS OF LETTERS SIMULTANEOUSLY",
    "BEAUFORT IS THE RECIPROCAL OF VIGENERE AND DECRYPTS ITSELF",
    "THE AUTOKEY CIPHER EXTENDS THE KEY WITH THE PLAINTEXT ITSELF",
    "GRONSFELD USES A DIGIT SEQUENCE AS THE REPEATING KEY",
    "THE TRITHEMIUS CIPHER ADVANCES THE ALPHABET BY ONE POSITION EACH LETTER",
    "A SCYTALE IS A ROD USED BY SPARTANS TO ENCRYPT MESSAGES",
    "DOUBLE TRANSPOSITION APPLIES COLUMNAR ENCRYPTION TWICE",
    "ROUTE CIPHERS READ A GRID IN A PRESCRIBED GEOMETRIC PATTERN",
    "POLYBIUS ENCODED LETTERS AS PAIRS OF NUMBERS FROM ONE TO FIVE",
    "THE BIFID CIPHER COMBINES A POLYBIUS SQUARE WITH TRANSPOSITION",
    "HILL CIPHER USES LINEAR ALGEBRA TO ENCRYPT BLOCKS OF LETTERS",
    "THE FOUR SQUARE CIPHER USES TWO KEYWORD POLYBIUS SQUARES",
    "TWO SQUARE CIPHER ENCRYPTS DIGRAPHS WITH TWO POLYBIUS ALPHABETS",
    "ERROR CORRECTION AND DETECTION ARE FUNDAMENTAL TO SECURE SYSTEMS",
    "THE FRIEDMAN TEST USES THE INDEX OF COINCIDENCE TO ESTIMATE KEY LENGTH",
    "KERCKHOFFS PRINCIPLE SAYS THE SYSTEM MUST BE SECURE EVEN IF ENEMY KNOWS ALGORITHM",
    "THE ONE TIME PAD IS THEORETICALLY UNBREAKABLE IF USED CORRECTLY",
    "DIFFIE AND HELLMAN INTRODUCED PUBLIC KEY CRYPTOGRAPHY IN NINETEEN SEVENTY SIX",
]


# ---------------------------------------------------------------------------
# Cipher implementations
# ---------------------------------------------------------------------------

def clean(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def with_spacing(s: str) -> str:
    if random.random() < 0.45:
        group = random.choice([4, 5])
        c = clean(s)
        return " ".join(c[i:i + group] for i in range(0, len(c), group))
    return s


def caesar(s: str, shift: int) -> str:
    out = []
    for ch in s.upper():
        if ch in ALPHABET:
            out.append(ALPHABET[(ALPHABET.index(ch) + shift) % 26])
        else:
            out.append(ch)
    return "".join(out)


def atbash(s: str) -> str:
    return s.upper().translate(str.maketrans(ALPHABET, ALPHABET[::-1]))


def vigenere(s: str, key: str) -> str:
    key = clean(key)
    out, j = [], 0
    for ch in s.upper():
        if ch in ALPHABET:
            k = ALPHABET.index(key[j % len(key)])
            out.append(ALPHABET[(ALPHABET.index(ch) + k) % 26])
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def beaufort(s: str, key: str) -> str:
    """Beaufort cipher: E(x) = (k - x) mod 26 — reciprocal."""
    key = clean(key)
    out, j = [], 0
    for ch in s.upper():
        if ch in ALPHABET:
            k = ALPHABET.index(key[j % len(key)])
            x = ALPHABET.index(ch)
            out.append(ALPHABET[(k - x) % 26])
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def gronsfeld(s: str, digit_key: str) -> str:
    """Gronsfeld cipher: Vigenère with digits (0–9) as key characters."""
    key = [int(d) for d in digit_key if d.isdigit()]
    if not key:
        key = [3, 1, 4, 1, 5]
    out, j = [], 0
    for ch in s.upper():
        if ch in ALPHABET:
            out.append(ALPHABET[(ALPHABET.index(ch) + key[j % len(key)]) % 26])
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def autokey(s: str, primer: str) -> str:
    """Autokey Vigenère: key starts with primer, then continues with plaintext."""
    chars = clean(s)
    primer = clean(primer)
    keystream = primer + chars
    out = []
    for i, ch in enumerate(chars):
        k = ALPHABET.index(keystream[i])
        out.append(ALPHABET[(ALPHABET.index(ch) + k) % 26])
    return "".join(out)


def trithemius(s: str) -> str:
    """Trithemius: progressive Caesar — shift increases by 1 per letter."""
    out, i = [], 0
    for ch in s.upper():
        if ch in ALPHABET:
            out.append(ALPHABET[(ALPHABET.index(ch) + i) % 26])
            i += 1
        else:
            out.append(ch)
    return "".join(out)


def porta(s: str, key: str) -> str:
    """Porta cipher (simplified): 13-alphabet polyalphabetic table."""
    # Each row shifts both halves; key letter selects which row (A/B→0, C/D→1, …)
    # Standard Porta mixes both halves of the alphabet in each row.
    PORTA_ROWS = [
        "NOPQRSTUVWXYZABCDEFGHIJKLM",
        "ONPQRSTUVWXYZABCDEFGHIJKLM",
        "OPNQRSTUVWXYZABCDEFGHIJKLM",  # simplified approximation
        "OPQNRSTUVWXYZABCDEFGHIJKLM",
        "OPQRNSTUVWXYZABCDEFGHIJKLM",
        "OPQRSNUVWXYZABCDEFGHIJKLMT",
        "OPQRSTUNVWXYZABCDEFGHIJKLM",
        "OPQRSTUVNWXYZABCDEFGHIJKLM",
        "OPQRSTUVWNXYZABCDEFGHIJKLM",
        "OPQRSTUVWXNYZABCDEFGHIJKLM",
        "OPQRSTUVWXYNZABCDEFGHIJKLM",
        "OPQRSTUVWXYZNABCDEFGHIJKLM",
        "OPQRSTUVWXYZANABCDEFGHIJKL",
    ]
    key = clean(key)
    out, j = [], 0
    for ch in s.upper():
        if ch in ALPHABET:
            ki = (ALPHABET.index(key[j % len(key)]) // 2) % 13
            row = PORTA_ROWS[ki]
            x = ALPHABET.index(ch)
            out.append(row[x])
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def rail_fence(text: str, rails: int) -> str:
    chars = clean(text)
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


def columnar(text: str, key: str) -> str:
    chars = clean(text)
    key = clean(key)
    cols = len(key)
    chars += "X" * ((-len(chars)) % cols)
    rows = [chars[i:i + cols] for i in range(0, len(chars), cols)]
    order = sorted(range(cols), key=lambda i: (key[i], i))
    return "".join("".join(row[i] for row in rows) for i in order)


def scytale(text: str, cols: int) -> str:
    """Scytale: wrap text around a rod of `cols` diameter and read off columns."""
    chars = clean(text)
    padding = (-len(chars)) % cols
    chars += "X" * padding
    n_rows = len(chars) // cols
    result = []
    for c in range(cols):
        for r in range(n_rows):
            result.append(chars[r * cols + c])
    return "".join(result)


def double_transposition(text: str, key1: str, key2: str) -> str:
    """Double columnar transposition: apply columnar encryption twice."""
    return columnar(columnar(text, key1), key2)


def stager_route(text: str, cols: int) -> str:
    """Route (Stager) cipher: fill rectangle row-by-row, read columns alternating direction."""
    chars = clean(text)
    padding = (-len(chars)) % cols
    chars += "X" * padding
    n_rows = len(chars) // cols
    grid = [chars[r * cols:(r + 1) * cols] for r in range(n_rows)]
    result = []
    for c in range(cols):
        if c % 2 == 0:
            for r in range(n_rows):
                result.append(grid[r][c])
        else:
            for r in range(n_rows - 1, -1, -1):
                result.append(grid[r][c])
    return "".join(result)


def affine(text: str, a: int, b: int) -> str:
    out = []
    for ch in text.upper():
        if ch in ALPHABET:
            out.append(ALPHABET[(a * ALPHABET.index(ch) + b) % 26])
        else:
            out.append(ch)
    return "".join(out)


def substitution(text: str, alphabet: str) -> str:
    trans = str.maketrans(ALPHABET, alphabet)
    return text.upper().translate(trans)


# ---------------------------------------------------------------------------
# Sparse-label cipher encoders (museum-only classes → synthetic generation)
# ---------------------------------------------------------------------------

# Bacon biliteral cipher: each letter → 5-bit A/B string
_BACON = {
    'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA',
    'F': 'AABAB', 'G': 'AABBA', 'H': 'AABBB', 'I': 'ABAAA', 'J': 'ABAAA',
    'K': 'ABAAB', 'L': 'ABABA', 'M': 'ABABB', 'N': 'ABBAA', 'O': 'ABBAB',
    'P': 'ABBBA', 'Q': 'ABBBB', 'R': 'BAAAA', 'S': 'BAAAB', 'T': 'BAABA',
    'U': 'BAABB', 'V': 'BAABB', 'W': 'BABAA', 'X': 'BABAB', 'Y': 'BABBA',
    'Z': 'BABBB',
}

def bacon_cipher_encode(text: str) -> str:
    return " ".join(_BACON.get(c, 'AAAAA') for c in clean(text))


# Polybius square: 5×5 grid (I=J), letters → row-col digit pair
_POLYBIUS_GRID: Dict[str, str] = {}
_p_idx = 0
for _ch in "ABCDEFGHIKLMNOPQRSTUVWXYZ":   # no J
    _POLYBIUS_GRID[_ch] = f"{_p_idx // 5 + 1}{_p_idx % 5 + 1}"
    _p_idx += 1
_POLYBIUS_GRID['J'] = _POLYBIUS_GRID['I']

def polybius_encode(text: str) -> str:
    return " ".join(_POLYBIUS_GRID.get(c, '11') for c in clean(text))


# Morse code: dots and dashes, word-separated by " / "
_MORSE: Dict[str, str] = {
    'A': '.-',   'B': '-...', 'C': '-.-.',  'D': '-..',  'E': '.',
    'F': '..-.',  'G': '--.',  'H': '....', 'I': '..',   'J': '.---',
    'K': '-.-',  'L': '.-..',  'M': '--',   'N': '-.',   'O': '---',
    'P': '.--.',  'Q': '--.-', 'R': '.-.',  'S': '...',  'T': '-',
    'U': '..-',  'V': '...-', 'W': '.--',  'X': '-..-', 'Y': '-.--',
    'Z': '--..',
}

def morse_encode(text: str) -> str:
    words = [re.sub(r'[^A-Z]', '', w) for w in text.upper().split()]
    return " / ".join(
        " ".join(_MORSE[c] for c in w if c in _MORSE)
        for w in words if w
    )


# Tap code: letters as row.col dot-groups separated by double-space
# 5×5 grid, K → C (no K row)
_TAP: Dict[str, tuple] = {}
_t_idx = 0
for _ch in "ABCDEFGHIJLMNOPQRSTUVWXYZ":   # no K
    _TAP[_ch] = (_t_idx // 5 + 1, _t_idx % 5 + 1)
    _t_idx += 1
_TAP['K'] = _TAP['C']

def tap_code_encode(text: str) -> str:
    parts = []
    for c in clean(text):
        row, col = _TAP.get(c, (1, 1))
        parts.append('.' * row + ' ' + '.' * col)
    return '  '.join(parts)


# Aeneas Tacticus: A=1 … Z=26 (space-separated integers)
def aeneas_tacticus_encode(text: str) -> str:
    return " ".join(str(ALPHABET.index(c) + 1) for c in clean(text))


# Wallis cipher: fixed shuffled 2-digit mapping (seeded for reproducibility)
_wallis_nums = list(range(10, 100))
random.Random(42).shuffle(_wallis_nums)
_WALLIS_MAP: Dict[str, str] = {c: str(_wallis_nums[i]) for i, c in enumerate(ALPHABET)}

def wallis_cipher_encode(text: str) -> str:
    return " ".join(_WALLIS_MAP[c] for c in clean(text))


# Homophonic substitution: common letters have multiple numeric homophones
_HOMO_TABLE: Dict[str, List[int]] = {}
_homo_pool = list(range(1, 100))
random.Random(7).shuffle(_homo_pool)
_homo_idx_ = 0
for _rank, _c in enumerate("ETAOINSHRDLCUMWFGYPBVKJXQZ"):
    _n = max(1, 4 - _rank // 6)   # E→4 homophones, Z→1
    _HOMO_TABLE[_c] = _homo_pool[_homo_idx_: _homo_idx_ + _n]
    _homo_idx_ = min(_homo_idx_ + _n, 98)

def homophonic_encode(text: str) -> str:
    return " ".join(str(random.choice(_HOMO_TABLE.get(c, [1]))) for c in clean(text))


# Culper Ring: ~3-digit numbers (800–998), 999 = word separator
def culper_ring_encode(text: str) -> str:
    groups: List[str] = []
    for word in text.upper().split():
        letters = re.sub(r'[^A-Z]', '', word)
        if not letters:
            continue
        for c in letters:
            groups.append(str(800 + ALPHABET.index(c) * 3 + random.randint(0, 2)))
        groups.append('999')
    return " ".join(groups)


# Zimmermann Telegram style: 4–5 digit numeric groups
def zimmermann_encode(text: str) -> str:
    codes: List[str] = []
    for c in clean(text):
        base = (ALPHABET.index(c) + 1) * 314 + random.randint(0, 99)
        if random.random() < 0.5:
            codes.append(f"{base % 10000:04d}")
        else:
            codes.append(f"{base % 100000:05d}")
    return " ".join(codes)


# Book cipher: page.line.word triples
def book_cipher_encode(text: str) -> str:
    parts = []
    for _c in clean(text):
        parts.append(f"{random.randint(1, 20)}.{random.randint(1, 50)}.{random.randint(1, 15)}")
    return " ".join(parts)


# Arnold-André book cipher: page.section.word triples (tighter ranges)
def arnold_andre_encode(text: str) -> str:
    parts = []
    for _c in clean(text):
        parts.append(f"{random.randint(1, 20)}.{random.randint(1, 6)}.{random.randint(1, 10)}")
    return " ".join(parts)


# Babington nomenclator: ⟨wNN⟩ / ⟨aNN⟩ tokens
_BABINGTON_MAP: Dict[str, str] = {
    c: f"⟨{'w' if i % 2 == 0 else 'a'}{(i * 3 + 7) % 40 + 1:02d}⟩"
    for i, c in enumerate(ALPHABET)
}

def babington_encode(text: str) -> str:
    return " ".join(_BABINGTON_MAP.get(c, '⟨w00⟩') for c in clean(text))


# Navajo Code Talker
_NAVAJO: Dict[str, str] = {
    'A': 'WOL-LA-CHEE', 'B': 'SHUSH',         'C': 'MOASI',          'D': 'BE',
    'E': 'DZEH',        'F': 'MA-E',            'G': 'AH-JAH',         'H': 'LIN',
    'I': 'TKIN',        'J': 'TKELE-CHO-G',     'K': 'KLIZZIE',        'L': 'DIBEH-YAZZIE',
    'M': 'NA-AS-TSO-SI','N': 'TSAH',            'O': 'A-KHA',          'P': 'CLA-GI-AIH',
    'Q': 'CA-YEILTH',   'R': 'GAH',             'S': 'DIBEH',          'T': 'THAN-ZIE',
    'U': 'NO-DA-IH',    'V': 'A-KEH-DI-GLINI',  'W': 'GLOE-IH',        'X': 'AL-AN-AS-DZOH',
    'Y': 'TSAH-AS-ZIH', 'Z': 'BESH-DO-TLIZ',
}

def navajo_encode(text: str) -> str:
    chars = clean(text)[:40]   # keep length sane
    return "   /   ".join(_NAVAJO.get(c, 'WOL-LA-CHEE') for c in chars)


# Null / acrostic cipher: first letter of each word encodes message
_ACROSTIC: Dict[str, List[str]] = {
    'A': ['again','along','after','above','alert','among','array'],
    'B': ['below','brief','board','broad','basis','batch'],
    'C': ['could','clear','cover','cause','cross','carry','claim'],
    'D': ['depot','delay','draft','drive','daily','depth'],
    'E': ['enemy','early','eight','extra','exact','enter'],
    'F': ['force','front','first','field','flank','final'],
    'G': ['guard','group','guide','given','going','gains'],
    'H': ['heavy','holds','heads','hence','hours','house'],
    'I': ['intel','inner','issue','input','incur'],
    'J': ['joint','joins','judge','jetty'],
    'K': ['keeps','knife','known','kings'],
    'L': ['light','large','lines','lower','leads','local'],
    'M': ['major','march','motor','mixed','miles','moved'],
    'N': ['north','night','naval','notes','needs'],
    'O': ['order','outer','often','opens','other'],
    'P': ['point','posts','place','parts','phase'],
    'Q': ['quick','quiet','quota'],
    'R': ['range','route','radio','raids','right','roads'],
    'S': ['south','swift','scout','stage','shore','seeks'],
    'T': ['troop','takes','three','track','turns'],
    'U': ['under','unite','until','upper','units'],
    'V': ['vital','vague','value','verse'],
    'W': ['where','while','width','wings','watch'],
    'X': ['xerox','extra'],
    'Y': ['yield','years','young'],
    'Z': ['zones','zonal','zeal'],
}

def null_cipher_encode(text: str) -> str:
    chars = clean(text)[:25]
    return " ".join(random.choice(_ACROSTIC.get(c, ['extra'])) for c in chars)


# Pigpen cipher: Unicode symbol representations (matching real dataset format)
_PIGPEN_SYMS = [
    '⌐', '¬', 'r', 'Γ', 'L', '⌐·', '¬·', 'r·', 'Γ·',
    '┘', '┐', '└', '┌', '─', '┘·', '┐·', '└·', '┌·',
    'X', 'K', 'X·', 'K·',
    '⌒', '⌣', '⌒·', '⌣·',
]
_PIGPEN_TABLE: Dict[str, str] = {ALPHABET[i]: _PIGPEN_SYMS[i % len(_PIGPEN_SYMS)] for i in range(26)}

def pigpen_encode(text: str) -> str:
    return " ".join(_PIGPEN_TABLE[c] for c in clean(text))


# Commercial code: 5-letter groups; many ending in Z
def commercial_code_encode(text: str) -> str:
    chars = clean(text)
    if not chars:
        chars = "HELLO"
    groups: List[str] = []
    for i in range(0, len(chars), 5):
        chunk = list(chars[i:i + 5])
        while len(chunk) < 5:
            chunk.append(random.choice('XZABCDE'))
        if random.random() < 0.4:
            chunk[-1] = random.choice('XZ')
        groups.append("".join(chunk))
    return " ".join(groups)


# Voynich manuscript render: pseudo-syllabic words
_VOYNICH_WORDS = [
    'oror', 'oxed', 'otol', 'qokeedy', 'chedy', 'shedy', 'dainy', 'otaiin',
    'cheol', 'ykcheol', 'qokal', 'raiin', 'aiiin', 'chol', 'chor',
    'daral', 'shal', 'sharal', 'okal', 'daiin', 'chedal', 'sheedy',
    'otedy', 'qotedy', 'ctaiin', 'ytedy', 'rcheol', 'okeedy', 'lkaiin',
    'okshy', 'okedy', 'otain', 'qokedy', 'okain', 'choldaiin', 'sheol',
    'ypchol', 'dal', 'shor', 'kaiiin', 'qokeedy', 'ytaiin', 'shaiin',
]

def voynich_render_encode(text: str) -> str:
    n_words = max(10, len(clean(text)) // 3)
    return " ".join(random.choice(_VOYNICH_WORDS) for _ in range(n_words))


# Running key: Vigenère with a full sentence as the key
def running_key_encode(text: str) -> str:
    plain = clean(text)
    key = clean(random.choice(BASE_SENTENCES))
    while len(key) < len(plain):
        key += key
    out = [ALPHABET[(ALPHABET.index(p) + ALPHABET.index(key[i])) % 26]
           for i, p in enumerate(plain)]
    return with_spacing("".join(out))


# One-time pad: truly random letter sequence
def one_time_pad_encode(text: str) -> str:
    return with_spacing("".join(random.choice(ALPHABET) for _ in clean(text)))


# Vernam: same structure as OTP for text representation
def vernam_encode(text: str) -> str:
    return with_spacing("".join(random.choice(ALPHABET) for _ in clean(text)))


def make_plain() -> str:
    k = random.randint(1, 4)
    parts = random.sample(BASE_SENTENCES, k=k)
    plain = " ".join(parts)
    if random.random() < 0.12:
        plain += " " + random.choice([
            "TODAY", "TOMORROW", "CAREFULLY", "HONESTLY", "SLOWLY",
            "ALWAYS", "NEVER", "CERTAINLY", "PERHAPS",
        ])
    return plain


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def record(
    label: str, plaintext: str, ciphertext: str, meta: Dict[str, object], rid: int
) -> Dict[str, object]:
    text_clean = clean(ciphertext)
    # Symbolic/numeric ciphers (tap code, morse, polybius, book ciphers, etc.) lose their
    # distinctive format when passed through with_spacing()'s clean() branch.
    # Preserve the raw ciphertext when it is not primarily alphabetic.
    cipher_nonspace = ciphertext.replace(" ", "")
    alpha_ratio = len(text_clean) / max(1, len(cipher_nonspace))
    if alpha_ratio >= 0.5:
        display_text = with_spacing(ciphertext)
    else:
        display_text = ciphertext   # keep dots / numbers / symbols intact
    # text_length uses the plaintext letter count for non-alphabetic ciphers
    # so difficulty/length are still meaningful even for numeric outputs.
    n = len(text_clean) if text_clean else len(clean(plaintext))
    if n < 30:
        difficulty = "easy"
    elif n < 80:
        difficulty = "medium"
    else:
        difficulty = "hard"
    return {
        "id": f"cda-{rid:07d}",
        "text": display_text,
        "ciphertext": display_text,
        "plaintext": plaintext,
        "label": label,
        "cipher": label,
        "key": meta,
        "difficulty": difficulty,
        "language": "en",
        "text_length": n,
        "length": n,
        "attack_methods": ATTACKS.get(label, []),
        "educational_note": EDU_NOTES.get(label, ""),
        "source": "synthetic_educational",
    }


# ---------------------------------------------------------------------------
# Per-label generator
# ---------------------------------------------------------------------------

_VIGENERE_KEYS = [
    "KEY", "LIBRARY", "CIPHER", "MUSEUM", "PRAYER", "SECURE", "PATTERN",
    "HONEST", "PUBLIC", "MODEL", "SIGNAL", "ENIGMA", "LIBERTY", "FREEDOM",
    "JUSTICE", "WARRIOR", "DRAGON", "THUNDER", "PALACE", "BATTLE",
]
_DOUBLE_KEYS_A = ["CIPHER", "MUSEUM", "SECURE", "LIBERTY", "JUSTICE"]
_DOUBLE_KEYS_B = ["PATTERN", "SIGNAL", "FREEDOM", "DRAGON", "PALACE"]
_GRONSFELD_KEYS = ["314159", "271828", "141421", "31416", "27183", "11235", "99999"]
_PORTA_KEYS = ["KEY", "LIBRARY", "CIPHER", "SECURE", "PATTERN", "JUSTICE"]


def build_row(label: str, rid: int) -> Dict[str, object]:  # noqa: C901
    plain = make_plain()

    if label in ("plaintext",):
        return record(label, plain, plain, {}, rid)

    if label in ("caesar_rot", "caesar"):
        shift = random.randint(1, 25)
        return record(label, plain, caesar(plain, shift), {"shift": shift}, rid)

    if label == "rot13":
        return record(label, plain, caesar(plain, 13), {"shift": 13}, rid)

    if label == "atbash":
        return record(label, plain, atbash(plain), {}, rid)

    if label == "vigenere":
        key = random.choice(_VIGENERE_KEYS)
        return record(label, plain, vigenere(plain, key), {"key": key}, rid)

    if label == "beaufort":
        key = random.choice(_VIGENERE_KEYS)
        return record(label, plain, beaufort(plain, key), {"key": key}, rid)

    if label == "gronsfeld":
        key = random.choice(_GRONSFELD_KEYS)
        return record(label, plain, gronsfeld(plain, key), {"key": key}, rid)

    if label == "autokey":
        primer = random.choice(_VIGENERE_KEYS[:8])
        return record(label, plain, autokey(plain, primer), {"primer": primer}, rid)

    if label == "trithemius":
        return record(label, plain, trithemius(plain), {}, rid)

    if label == "porta":
        key = random.choice(_PORTA_KEYS)
        return record(label, plain, porta(plain, key), {"key": key}, rid)

    if label == "rail_fence":
        rails = random.choice([2, 3, 4, 5])
        return record(label, plain, rail_fence(plain, rails), {"rails": rails}, rid)

    if label in ("columnar", "columnar_transposition"):
        key = random.choice(_VIGENERE_KEYS)
        return record(label, plain, columnar(plain, key), {"key": key}, rid)

    if label == "scytale":
        cols = random.randint(4, 12)
        return record(label, plain, scytale(plain, cols), {"cols": cols}, rid)

    if label == "double_transposition":
        key1 = random.choice(_DOUBLE_KEYS_A)
        key2 = random.choice(_DOUBLE_KEYS_B)
        return record(label, plain, double_transposition(plain, key1, key2),
                      {"key1": key1, "key2": key2}, rid)

    if label == "stager_route":
        cols = random.randint(5, 12)
        return record(label, plain, stager_route(plain, cols), {"cols": cols}, rid)

    if label == "affine":
        a, b = random.choice(AFFINE_A_VALUES), random.randint(0, 25)
        return record(label, plain, affine(plain, a, b), {"a": a, "b": b}, rid)

    if label in ("substitution", "monoalphabetic"):
        alph = list(ALPHABET)
        random.shuffle(alph)
        alph = "".join(alph)
        return record(label, plain, substitution(plain, alph), {"alphabet": alph}, rid)

    # --- Sparse museum-label generators ---

    if label == "bacon_cipher":
        return record(label, plain, bacon_cipher_encode(plain), {}, rid)

    if label == "polybius":
        return record(label, plain, polybius_encode(plain), {}, rid)

    if label == "morse_code":
        return record(label, plain, morse_encode(plain), {}, rid)

    if label == "tap_code":
        return record(label, plain, tap_code_encode(plain), {}, rid)

    if label == "aeneas_tacticus":
        return record(label, plain, aeneas_tacticus_encode(plain), {}, rid)

    if label == "wallis_cipher":
        return record(label, plain, wallis_cipher_encode(plain), {}, rid)

    if label == "homophonic":
        return record(label, plain, homophonic_encode(plain), {}, rid)

    if label == "culper_ring":
        return record(label, plain, culper_ring_encode(plain), {}, rid)

    if label == "zimmermann":
        return record(label, plain, zimmermann_encode(plain), {}, rid)

    if label == "book_cipher":
        return record(label, plain, book_cipher_encode(plain), {}, rid)

    if label == "arnold_andre":
        return record(label, plain, arnold_andre_encode(plain), {}, rid)

    if label == "babington":
        return record(label, plain, babington_encode(plain), {}, rid)

    if label == "navajo_code":
        return record(label, plain, navajo_encode(plain), {}, rid)

    if label == "null_cipher":
        return record(label, plain, null_cipher_encode(plain), {}, rid)

    if label == "pigpen":
        return record(label, plain, pigpen_encode(plain), {}, rid)

    if label == "commercial_code":
        return record(label, plain, commercial_code_encode(plain), {}, rid)

    if label == "voynich_render":
        return record(label, plain, voynich_render_encode(plain), {}, rid)

    if label == "running_key":
        return record(label, plain, running_key_encode(plain), {}, rid)

    if label == "one_time_pad":
        return record(label, plain, one_time_pad_encode(plain), {}, rid)

    if label == "vernam":
        return record(label, plain, vernam_encode(plain), {}, rid)

    raise ValueError(f"Unknown label: {label!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Labels the synthetic generator can produce.  Keep in sync with build_row().
_SYNTH_LABELS = [
    "plaintext",
    "caesar_rot", "caesar", "rot13", "atbash", "affine",
    "substitution", "monoalphabetic",
    "vigenere", "beaufort", "gronsfeld", "autokey", "trithemius", "porta",
    "rail_fence", "columnar", "columnar_transposition",
    "scytale", "double_transposition", "stager_route",
    # sparse museum-only labels now with synthetic generators
    "aeneas_tacticus", "arnold_andre", "babington", "bacon_cipher",
    "book_cipher", "commercial_code", "culper_ring", "homophonic",
    "morse_code", "navajo_code", "null_cipher", "one_time_pad",
    "pigpen", "polybius", "running_key", "tap_code",
    "vernam", "voynich_render", "wallis_cipher", "zimmermann",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate or extend the synthetic cipher-examples dataset."
    )
    ap.add_argument("--out", default="data/cipher_examples.jsonl")
    ap.add_argument("--n", type=int, default=10000,
                    help="Number of synthetic examples to generate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--append", action="store_true",
        help="Append to existing file instead of overwriting. "
             "Uses max existing id to avoid collisions.",
    )
    ap.add_argument(
        "--labels", nargs="*", default=None,
        help="Restrict generation to these labels (default: all synthetic labels).",
    )
    ap.add_argument(
        "--per-label", type=int, default=None,
        help="Generate exactly this many examples per label (overrides --n).",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    labels = args.labels or _SYNTH_LABELS
    unknown = [l for l in labels if l not in _SYNTH_LABELS]
    if unknown:
        ap.error(f"Unknown labels (not implemented in synthetic generator): {unknown}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Determine starting row ID to avoid collisions when appending.
    start_id = 0
    if args.append and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rid = int(json.loads(line).get("id", "cda-0").split("-")[-1])
                    start_id = max(start_id, rid + 1)
                except (ValueError, KeyError):
                    pass

    if args.per_label is not None:
        total = args.per_label * len(labels)
        plan = [lbl for lbl in labels for _ in range(args.per_label)]
        random.shuffle(plan)
    else:
        total = args.n
        plan = [labels[i % len(labels)] for i in range(total)]
        random.shuffle(plan)

    mode = "a" if args.append else "w"
    written = 0
    with out.open(mode, encoding="utf-8") as f:
        for i, label in enumerate(plan):
            row = build_row(label, start_id + i)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written:,} examples ({'appended' if args.append else 'created'}) → {out}")


if __name__ == "__main__":
    main()
