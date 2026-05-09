from __future__ import annotations

import argparse
import json
import random
import re
import string
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
    text_with_spacing = with_spacing(ciphertext)
    n = len(text_clean)
    if n < 30:
        difficulty = "easy"
    elif n < 80:
        difficulty = "medium"
    else:
        difficulty = "hard"
    return {
        "id": f"cda-{rid:07d}",
        "text": text_with_spacing,
        "ciphertext": text_with_spacing,
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


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
AFFINE_A_VALUES = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]

ATTACKS: Dict[str, List[str]] = {
    "plaintext":   [],
    "caesar_rot":  ["brute_force_26", "frequency_analysis", "chi_squared_english"],
    "atbash":      ["self_inverse_check", "frequency_analysis"],
    "vigenere":    ["kasiski_examination", "friedman_test", "index_of_coincidence", "frequency_per_column"],
    "rail_fence":  ["rail_count_search", "anagram_recovery"],
    "columnar":    ["key_length_search", "anagram_recovery"],
    "affine":      ["brute_force_312", "frequency_analysis"],
    "substitution":["frequency_analysis", "pattern_words", "hill_climbing_lm"],
}

EDU_NOTES: Dict[str, str] = {
    "plaintext":   "No cipher applied. Useful as a negative-class baseline.",
    "caesar_rot":  "Caesar / ROT-N is a single-shift monoalphabetic cipher with only 26 keys.",
    "atbash":      "Atbash maps A↔Z, B↔Y, … and is its own inverse — no key needed.",
    "vigenere":    "Vigenère uses a repeating-key Caesar shift; vulnerable to Kasiski and Friedman analysis.",
    "rail_fence":  "Rail-fence is a transposition cipher: it rearranges letters along a zig-zag pattern.",
    "columnar":    "Columnar transposition writes plaintext into rows and reads columns in key order.",
    "affine":      "Affine cipher: E(x) = (a·x + b) mod 26 with a coprime to 26 — only 312 keys.",
    "substitution":"Monoalphabetic substitution with an arbitrary 26-letter permutation.",
}

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
    # Additional variety — more diverse vocabulary improves classifier generalization.
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
]

def clean(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())

def with_spacing(s: str) -> str:
    # Occasionally group text to mimic old ciphertext formatting.
    if random.random() < 0.45:
        group = random.choice([4, 5])
        c = clean(s)
        return " ".join(c[i:i+group] for i in range(0, len(c), group))
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
    rows = [chars[i:i+cols] for i in range(0, len(chars), cols)]
    order = sorted(range(cols), key=lambda i: (key[i], i))
    return "".join("".join(row[i] for row in rows) for i in order)

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

def make_plain() -> str:
    k = random.randint(1, 4)
    parts = random.sample(BASE_SENTENCES, k=k)
    plain = " ".join(parts)
    # Add mild variation while keeping labels reproducible.
    if random.random() < 0.15:
        plain += " " + random.choice(["TODAY", "TOMORROW", "CAREFULLY", "HONESTLY", "SLOWLY"])
    return plain

if __name__ == "__main__":
    main()
