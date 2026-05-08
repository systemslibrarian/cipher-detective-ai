"""Tests for Cipher Detective AI's core cryptanalysis utilities."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import (
    affine_decrypt,
    affine_encrypt,
    atbash,
    best_affine_candidates,
    best_caesar_candidates,
    caesar_encrypt,
    caesar_shift,
    chi_squared_for_english,
    clean_letters,
    columnar_transposition_encrypt,
    friedman_key_length,
    heuristic_classify,
    index_of_coincidence,
    kasiski_key_lengths,
    rail_fence_encrypt,
    shannon_entropy,
    substitution_encrypt,
    transposition_signal,
    vigenere_decrypt,
    vigenere_encrypt,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Cipher encoders / decoders
# ---------------------------------------------------------------------------

def test_clean_letters_basic():
    assert clean_letters("A b-c! 123") == "ABC"


def test_clean_letters_empty_and_nonalpha():
    assert clean_letters("") == ""
    assert clean_letters("12345 !@#$%") == ""


def test_caesar_round_trip():
    msg = "THIS IS A TEST"
    assert caesar_shift(caesar_encrypt(msg, 3), 3) == msg


def test_caesar_preserves_punctuation():
    assert caesar_encrypt("HELLO, WORLD!", 1) == "IFMMP, XPSME!"


def test_atbash_self_inverse():
    msg = "ATTACK AT DAWN"
    assert atbash(atbash(msg)) == msg
    assert atbash("GSV") == "THE"


def test_affine_round_trip():
    msg = "AFFINE CIPHER"
    enc = affine_encrypt(msg, 5, 8)
    assert affine_decrypt(enc, 5, 8) == msg


def test_affine_invalid_a_raises():
    with pytest.raises(ValueError):
        affine_decrypt("HELLO", 2, 3)  # gcd(2, 26) != 1


def test_vigenere_known_example():
    assert vigenere_encrypt("ATTACKATDAWN", "LEMON") == "LXFOPVEFRNHR"


def test_vigenere_round_trip():
    msg = "MEET ME AT MIDNIGHT BY THE OLD OAK"
    assert vigenere_decrypt(vigenere_encrypt(msg, "MUSEUM"), "MUSEUM") == msg


def test_vigenere_empty_key_raises():
    with pytest.raises(ValueError):
        vigenere_encrypt("HELLO", "")


def test_rail_fence_alpha_only():
    assert rail_fence_encrypt("WE ARE DISCOVERED", 3).isalpha()


def test_rail_fence_one_rail_is_identity():
    assert rail_fence_encrypt("HELLO WORLD", 1) == "HELLOWORLD"


def test_columnar_alpha_only():
    assert columnar_transposition_encrypt("WE ARE DISCOVERED", "KEY").isalpha()


def test_substitution_is_permutation():
    mapping = "QWERTYUIOPASDFGHJKLZXCVBNM"
    out = substitution_encrypt("HELLO", mapping)
    assert len(out) == 5 and out.isalpha()


def test_substitution_invalid_mapping_raises():
    with pytest.raises(ValueError):
        substitution_encrypt("HELLO", "ABC")  # not a 26-letter perm


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_ioc_uniform_letters_is_one():
    assert index_of_coincidence("AAAAAA") == 1.0


def test_ioc_random_text_near_baseline():
    # All 26 letters, evenly distributed -> ~1/26.
    s = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4)
    assert 0.030 <= index_of_coincidence(s) <= 0.045


def test_entropy_zero_on_constant_text():
    assert shannon_entropy("AAAAA") == 0.0


def test_entropy_short_circuit_empty():
    assert shannon_entropy("") == 0.0


def test_chi_squared_finite_on_english():
    val = chi_squared_for_english(clean_letters("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"))
    assert val < 200.0


def test_kasiski_finds_known_period():
    # Vigenère with a 6-letter key on long English text should leave repeats.
    plain = "THIS IS A LONG ENOUGH SAMPLE TO LET KASISKI FIND REPEATS " * 4
    ct = clean_letters(vigenere_encrypt(plain, "MUSEUM"))
    candidates = kasiski_key_lengths(ct)
    # Either 6 or a multiple of 6 should be present.
    keys = [k for k, _ in candidates]
    assert any(k % 6 == 0 for k in keys), f"got {candidates}"


def test_friedman_estimate_shape():
    plain = "THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY " * 6
    ct = clean_letters(vigenere_encrypt(plain, "CIPHER"))
    est = friedman_key_length(ct)
    assert 2 <= est <= 12


def test_transposition_signal_high_for_rail_fence():
    plain = "THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY AND TEACHES " * 3
    ct = clean_letters(rail_fence_encrypt(plain, 4))
    transp, bg = transposition_signal(ct)
    assert transp > 0.3
    assert bg < 0.6


def test_best_caesar_finds_correct_shift():
    plain = "THE DETECTIVE STUDIES PATTERNS BEFORE MAKING CLAIMS"
    ct = caesar_encrypt(plain, 7)
    cands = best_caesar_candidates(ct, top_n=3)
    assert cands[0][0] == 7


def test_best_affine_finds_correct_key():
    plain = "FREQUENCY ANALYSIS CAN REVEAL WEAK CIPHERS"
    ct = affine_encrypt(plain, 5, 8)
    cands = best_affine_candidates(ct, top_n=3)
    assert (cands[0][0], cands[0][1]) == (5, 8)


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

def test_heuristic_returns_scores():
    pred = heuristic_classify("WKLV LV D FDHVDU FLSKHU GHPR IRU FLSKHU GHWHFWLYH")
    assert pred.label in pred.scores
    assert pred.source == "heuristic"


def test_heuristic_handles_empty_input():
    pred = heuristic_classify("")
    assert pred.label == "too_short"


def test_heuristic_handles_nonalpha_input():
    pred = heuristic_classify("12345 !@#$% ^&*()")
    assert pred.label == "too_short"


def test_heuristic_caesar_label():
    plain = "THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY"
    pred = heuristic_classify(caesar_encrypt(plain, 5))
    assert pred.label == "caesar_rot"


def test_heuristic_atbash_label():
    plain = "FREQUENCY ANALYSIS CAN REVEAL WEAK CIPHERS"
    pred = heuristic_classify(atbash(plain))
    assert pred.label == "atbash"


def test_heuristic_plaintext_label():
    pred = heuristic_classify("THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY")
    assert pred.label == "plaintext"


def test_heuristic_short_sample_marked_uncertain():
    pred = heuristic_classify("ABCDE")
    assert pred.confidence <= 0.30


# ---------------------------------------------------------------------------
# Dataset generator schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "id", "text", "ciphertext", "plaintext", "label", "cipher", "key",
    "difficulty", "language", "text_length", "length", "attack_methods",
    "educational_note", "source",
}
ALLOWED_LABELS = {
    "plaintext", "caesar_rot", "atbash", "vigenere",
    "rail_fence", "columnar", "affine", "substitution",
}


def test_dataset_generator_schema(tmp_path):
    out = tmp_path / "tiny.jsonl"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_dataset.py"),
        "--out", str(out),
        "--n", "32",
        "--seed", "123",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, res.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 32
    for r in rows:
        assert REQUIRED_KEYS.issubset(r.keys()), f"missing keys: {REQUIRED_KEYS - r.keys()}"
        assert r["label"] in ALLOWED_LABELS
        assert r["cipher"] == r["label"]
        assert r["language"] == "en"
        assert r["text_length"] == r["length"] >= 0
        assert r["difficulty"] in {"easy", "medium", "hard"}
        assert isinstance(r["attack_methods"], list)
        assert isinstance(r["educational_note"], str)


def test_dataset_generator_seed_reproducible(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    base = [sys.executable, str(REPO_ROOT / "scripts" / "generate_dataset.py"), "--n", "16", "--seed", "7"]
    subprocess.run(base + ["--out", str(a)], check=True, cwd=REPO_ROOT)
    subprocess.run(base + ["--out", str(b)], check=True, cwd=REPO_ROOT)
    assert a.read_text() == b.read_text()

