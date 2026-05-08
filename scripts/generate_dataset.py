from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path
from typing import Dict, Iterable, List

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

def record(label: str, plaintext: str, ciphertext: str, meta: Dict[str, object], rid: int) -> Dict[str, object]:
    text_clean = clean(ciphertext)
    text_with_spacing = with_spacing(ciphertext)
    if len(text_clean) < 30:
        difficulty = "easy"
    elif len(text_clean) < 80:
        difficulty = "medium"
    else:
        difficulty = "hard"
    attack_methods = ATTACKS.get(label, [])
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
        "text_length": len(text_clean),
        "length": len(text_clean),
        "attack_methods": attack_methods,
        "educational_note": EDU_NOTES.get(label, ""),
        "source": "synthetic_educational",
    }

def build_row(label: str, keys: List[str], rid: int) -> Dict[str, object]:
    plain = make_plain()
    if label == "plaintext":
        return record(label, plain, plain, {}, rid)
    if label == "caesar_rot":
        shift = random.randint(1, 25)
        return record(label, plain, caesar(plain, shift), {"shift": shift}, rid)
    if label == "atbash":
        return record(label, plain, atbash(plain), {}, rid)
    if label == "vigenere":
        key = random.choice(keys)
        return record(label, plain, vigenere(plain, key), {"key": key}, rid)
    if label == "rail_fence":
        rails = random.choice([2, 3, 4, 5])
        return record(label, plain, rail_fence(plain, rails), {"rails": rails}, rid)
    if label == "columnar":
        key = random.choice(keys)
        return record(label, plain, columnar(plain, key), {"key": key}, rid)
    if label == "affine":
        a, b = random.choice(AFFINE_A_VALUES), random.randint(0, 25)
        return record(label, plain, affine(plain, a, b), {"a": a, "b": b}, rid)
    if label == "substitution":
        alphabet = list(ALPHABET)
        random.shuffle(alphabet)
        alphabet = "".join(alphabet)
        return record(label, plain, substitution(plain, alphabet), {"alphabet": alphabet}, rid)
    raise ValueError(label)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cipher_examples.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    labels = ["plaintext", "caesar_rot", "atbash", "vigenere", "rail_fence", "columnar", "affine", "substitution"]
    keys = ["KEY", "LIBRARY", "CIPHER", "MUSEUM", "PRAYER", "SECURE", "PATTERN", "HONEST", "PUBLIC", "MODEL"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            label = labels[i % len(labels)]
            if i >= len(labels):
                label = random.choice(labels)
            f.write(json.dumps(build_row(label, keys, i), ensure_ascii=False) + "\n")

    print(f"Wrote {args.n:,} examples to {out}")

if __name__ == "__main__":
    main()
