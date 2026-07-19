"""End-to-end benchmark of the auto-solvers: given ciphertext of a known type,
how often does each solver actually recover the plaintext?

Usage:
    python scripts/benchmark_solvers.py [--trials 30] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from core import (
    ALPHABET,
    affine_encrypt,
    best_affine_candidates,
    best_caesar_candidates,
    best_rail_fence_candidates,
    caesar_encrypt,
    clean_letters,
    hill_climb_substitution,
    rail_fence_encrypt,
    substitution_encrypt,
    vigenere_auto_solve,
    vigenere_encrypt,
)

SENTENCES = [
    "THE HISTORY OF CRYPTOGRAPHY IS FULL OF CLEVER IDEAS AND SPECTACULAR FAILURES THAT SHAPED MODERN SECURITY",
    "MEET ME AT THE OLD LIBRARY AFTER MIDNIGHT AND BRING THE DOCUMENTS WITH YOU",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG WHILE THE FARMER WATCHES FROM THE BARN",
    "GENERAL ORDERS REQUIRE ALL UNITS TO ADVANCE AT DAWN TOWARD THE EASTERN BRIDGE",
    "SCIENCE ADVANCES ONE CAREFUL EXPERIMENT AT A TIME AND EVERY FAILURE TEACHES SOMETHING NEW",
    "THE MUSEUM EXHIBIT OPENS NEXT WEEK AND FEATURES CIPHER MACHINES FROM THE SECOND WORLD WAR",
    "PLEASE DELIVER THE PACKAGE TO THE HARBOR MASTER BEFORE THE EVENING TIDE ARRIVES",
    "KNOWLEDGE OF LETTER FREQUENCIES ALLOWS ANALYSTS TO BREAK SIMPLE SUBSTITUTION QUICKLY",
]
KEYS = ["FORTRESS", "LEMON", "CIPHER", "AUTUMN", "LIBRARY", "SIGNAL"]


def truncate(s: str, n: int) -> str:
    return clean_letters(s)[:n]


def bench(trials: int, seed: int):
    rng = random.Random(seed)
    lengths = [40, 80, 160, 300]
    results: dict[str, dict[int, list[int]]] = {}

    def record(solver: str, n: int, ok: bool):
        results.setdefault(solver, {}).setdefault(n, []).append(int(ok))

    for _ in range(trials):
        # Diverse natural text: concatenated distinct sentences.  (Repeating
        # one sentence to reach the target length produces pathological n-gram
        # statistics that no frequency-based method should be judged on.)
        pool = SENTENCES[:]
        rng.shuffle(pool)
        sent = " ".join(pool)
        for n in lengths:
            pt = truncate(sent, n)

            shift = rng.randrange(1, 26)
            ct = caesar_encrypt(pt, shift)
            cands = best_caesar_candidates(ct, 1)
            record("caesar_auto", n, bool(cands) and cands[0][3] == pt)

            a = rng.choice([3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25])
            b = rng.randrange(26)
            ct = affine_encrypt(pt, a, b)
            acands = best_affine_candidates(ct, 1)
            record("affine_auto", n, bool(acands) and acands[0][4] == pt)

            key = rng.choice(KEYS)
            ct = vigenere_encrypt(pt, key)
            sols = vigenere_auto_solve(ct)
            got = clean_letters(sols[0][1]) if sols else ""
            record("vigenere_auto", n, got == pt)

            rails = rng.randrange(2, 8)
            ct = rail_fence_encrypt(pt, rails)
            rcands = best_rail_fence_candidates(ct, max_rails=10)
            record("railfence_auto", n, bool(rcands) and rcands[0][1] == pt)

            if n >= 160:  # hill climb needs length; only bench realistic sizes
                mapping = list(ALPHABET)
                rng.shuffle(mapping)
                ct = substitution_encrypt(pt, "".join(mapping))
                decoded, _key, _score = hill_climb_substitution(ct)
                decoded = clean_letters(decoded)
                # Success = ≥ 90% of letters correct (hill climb converges close)
                correct = sum(p == q for p, q in zip(decoded, pt, strict=False)) / len(pt)
                record("substitution_hillclimb", n, correct >= 0.90)

    print(f"{'solver':24s}" + "".join(f"  n={n:<5d}" for n in lengths))
    for solver, by_n in results.items():
        row = f"{solver:24s}"
        for n in lengths:
            if n in by_n:
                vals = by_n[n]
                row += f"  {100*sum(vals)/len(vals):5.1f}%"
            else:
                row += "      —"
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    bench(args.trials, args.seed)
