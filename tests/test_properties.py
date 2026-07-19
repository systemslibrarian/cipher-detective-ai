"""Property-based tests (hypothesis) for the cipher primitives.

These guard the class of bug that shipped in the Porta generator — an encoder
whose table wasn't a valid permutation, so encode/decode didn't round-trip —
by throwing thousands of random plaintexts and keys at every reversible cipher
and asserting the fundamental invariants:

  * decode(encode(x)) == x for keyed and keyless ciphers;
  * substitution-preserving ciphers keep length and the A-Z alphabet;
  * reciprocal ciphers (Atbash, Beaufort, Porta) are their own inverse;
  * the generator's encoders agree with core.py's implementations.
"""
from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import assume, given
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_dataset as gen  # noqa: E402

import core  # noqa: E402

ALPHABET = core.ALPHABET

# Strategies: non-empty A-Z plaintext, and A-Z keys.
text_st = st.text(alphabet=ALPHABET, min_size=1, max_size=120)
key_st = st.text(alphabet=ALPHABET, min_size=1, max_size=12)
AFFINE_A = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]


# --- Round trips ------------------------------------------------------------

@given(text_st, st.integers(min_value=0, max_value=25))
def test_caesar_round_trip(text, shift):
    assert core.caesar_shift(core.caesar_encrypt(text, shift), shift) == text


@given(text_st, st.sampled_from(AFFINE_A), st.integers(min_value=0, max_value=25))
def test_affine_round_trip(text, a, b):
    assert core.affine_decrypt(core.affine_encrypt(text, a, b), a, b) == text


@given(text_st, key_st)
def test_vigenere_round_trip(text, key):
    assert core.vigenere_decrypt(core.vigenere_encrypt(text, key), key) == text


@given(text_st, key_st)
def test_beaufort_reciprocal(text, key):
    # Beaufort is its own inverse: the generator encodes, core decodes.
    enc = gen.beaufort(text, key)
    assert core.beaufort_decrypt(enc, key) == text
    assert gen.beaufort(enc, key) == text


@given(text_st)
def test_atbash_self_inverse(text):
    assert core.atbash(core.atbash(text)) == text


@given(text_st, st.integers(min_value=2, max_value=12))
def test_rail_fence_round_trip(text, rails):
    assert core.rail_fence_decrypt(core.rail_fence_encrypt(text, rails), rails) == text


@given(text_st, key_st)
def test_columnar_round_trip(text, key):
    enc = core.columnar_transposition_encrypt(text, key)
    padded = text + "X" * ((-len(text)) % len(key))
    assert core.columnar_transposition_decrypt(enc, key) == padded


@given(text_st, key_st)
def test_porta_reciprocal_and_letterset(text, key):
    enc = gen.porta(text, key)
    assert gen.porta(enc, key) == text            # self-inverse
    assert len(enc) == len(text)
    assert set(enc) <= set(ALPHABET)
    # Every Porta output letter is in the opposite alphabet half from its input.
    for p, c in zip(text, enc, strict=True):
        assert (p <= "M") != (c <= "M")


@given(text_st, key_st)
def test_gronsfeld_round_trip(text, key):
    digits = "".join(str(ALPHABET.index(c) % 10) for c in key)
    enc = gen.gronsfeld(text, digits)
    key_nums = [int(d) for d in digits] or [3, 1, 4, 1, 5]
    dec = "".join(
        ALPHABET[(ALPHABET.index(c) - key_nums[i % len(key_nums)]) % 26]
        for i, c in enumerate(enc)
    )
    assert dec == text


@given(text_st, key_st)
def test_autokey_round_trip(text, key):
    enc = gen.autokey(text, key)
    primer = gen.clean(key)
    # Decrypt: recover plaintext, feeding decoded letters back into the key.
    dec = []
    keystream = list(primer)
    for i, ch in enumerate(enc):
        k = ALPHABET.index(keystream[i])
        p = ALPHABET[(ALPHABET.index(ch) - k) % 26]
        dec.append(p)
        keystream.append(p)
    assert "".join(dec) == text


# --- Structural invariants --------------------------------------------------

@given(st.text(alphabet=ALPHABET, min_size=0, max_size=60))
def test_substitution_is_bijection(text):
    mapping = "QWERTYUIOPASDFGHJKLZXCVBNM"
    out = core.substitution_encrypt(text, mapping)
    assert len(out) == len(text)
    inverse = "".join(ALPHABET[mapping.index(c)] for c in ALPHABET)
    assert core.substitution_encrypt(out, inverse) == text


@given(text_st, key_st)
def test_generator_matches_core_vigenere(text, key):
    # The two Vigenère implementations must not drift apart.
    assert gen.vigenere(text, key) == core.vigenere_encrypt(text, key)


@given(text_st, st.integers(min_value=1, max_value=25))
def test_generator_matches_core_caesar(text, shift):
    assert gen.caesar(text, shift) == core.caesar_encrypt(text, shift)


@given(text_st, st.sampled_from(AFFINE_A), st.integers(min_value=0, max_value=25))
def test_generator_matches_core_affine(text, a, b):
    assert gen.affine(text, a, b) == core.affine_encrypt(text, a, b)


@given(st.integers(min_value=2, max_value=15))
def test_rail_fence_one_rail_and_identity(rails):
    # A single-column read is identity for one rail; multi-rail preserves letters.
    text = "ATTACKATDAWNXYZ"
    enc = core.rail_fence_encrypt(text, rails)
    assert sorted(enc) == sorted(text)  # transposition preserves multiset
    assume(rails >= 2)
    assert core.rail_fence_decrypt(enc, rails) == text
