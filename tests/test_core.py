from core import (
    affine_decrypt,
    affine_encrypt,
    atbash,
    caesar_encrypt,
    caesar_shift,
    clean_letters,
    columnar_transposition_encrypt,
    heuristic_classify,
    index_of_coincidence,
    rail_fence_encrypt,
    vigenere_encrypt,
)

def test_clean_letters():
    assert clean_letters("A b-c! 123") == "ABC"

def test_caesar_round_trip():
    msg = "THIS IS A TEST"
    assert caesar_shift(caesar_encrypt(msg, 3), 3) == msg

def test_atbash():
    assert atbash("GSV") == "THE"

def test_affine_round_trip():
    msg = "AFFINE CIPHER"
    enc = affine_encrypt(msg, 5, 8)
    assert affine_decrypt(enc, 5, 8) == msg

def test_ioc_positive():
    assert index_of_coincidence("AAAA") == 1.0

def test_vigenere_known_example():
    assert vigenere_encrypt("ATTACKATDAWN", "LEMON") == "LXFOPVEFRNHR"

def test_transposition_outputs_letters():
    assert rail_fence_encrypt("WE ARE DISCOVERED", 3).isalpha()
    assert columnar_transposition_encrypt("WE ARE DISCOVERED", "KEY").isalpha()

def test_heuristic_returns_scores():
    pred = heuristic_classify("WKLV LV D FDHVDU FLSKHU GHPR IRU FLSKHU GHWHFWLYH")
    assert pred.label in pred.scores
    assert pred.source == "heuristic"
