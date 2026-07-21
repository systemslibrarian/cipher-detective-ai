"""
Cipher Detective AI — Hugging Face-native educational cryptanalysis Space.

This Space is built to be more than a generic hosted app:
- it can call a trained Hugging Face Transformer classifier;
- it always shows transparent classical cryptanalysis evidence;
- it includes challenge and explain modes for teaching.

Educational use only. Classical ciphers are weak by modern standards. This tool
does not break modern encryption or assist unauthorized access.
"""
from __future__ import annotations

import json
import os
import random

import gradio as gr

from core import (
    ModelPrediction,
    affine_decrypt,
    affine_encrypt,
    analyze_evidence,
    atbash,
    beaufort_auto_solve,
    beaufort_decrypt,
    best_affine_candidates,
    best_caesar_candidates,
    best_rail_fence_candidates,
    build_explanation,
    caesar_encrypt,
    caesar_shift,
    char_tokenize_text,
    chi_squared_for_english,
    clean_letters,
    columnar_auto_solve,
    columnar_transposition_decrypt,
    columnar_transposition_encrypt,
    english_quadgram_score,
    heuristic_classify,
    hill_climb_substitution,
    label_family,
    rail_fence_encrypt,
    shannon_entropy,
    substitution_encrypt,
    vigenere_auto_solve,
    vigenere_decrypt,
    vigenere_encrypt,
    word_score,
)

MODEL = None
MODEL_LABELS = None
MODEL_ERROR = None
MODEL_CHAR_LEVEL = False
TRANSFORMER_LAST_ERROR = None

_BASE_TOKENIZERS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}


def _load_transformer_calibration() -> list[tuple[float, float]]:
    """Isotonic confidence->accuracy breakpoints for the model, or [] if absent
    (see scripts/calibrate_transformer.py)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_calibration_map.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    return sorted((float(k), float(v)) for k, v in
                  data.get("fine_confidence_to_accuracy", {}).items())


_TRANSFORMER_CAL = _load_transformer_calibration()


def calibrate_transformer_confidence(raw: float) -> float:
    """Map a raw model softmax score to its empirical accuracy (piecewise-linear
    over the calibration curve). Identity if no map is present."""
    pts = _TRANSFORMER_CAL
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

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    model_id = os.getenv("CIPHER_MODEL_ID", "cipher_model")
    if os.path.isdir(model_id) or "/" in model_id:
        _model = AutoModelForSequenceClassification.from_pretrained(model_id)
        _tok = AutoTokenizer.from_pretrained(model_id)
        # Robustness: a mismatched/stale tokenizer (e.g. one carrying RoBERTa
        # special tokens on a BERT model) can emit token ids beyond the model's
        # embedding table, which crashes inference with an IndexError. Probe it,
        # and if it overflows, fall back to the clean base tokenizer for this
        # model type — so the classifier works regardless of the repo's or the
        # Space cache's tokenizer state.
        _vocab = _model.get_input_embeddings().weight.shape[0]
        if max(_tok("probe test", truncation=True, max_length=8)["input_ids"]) >= _vocab:
            _base = _BASE_TOKENIZERS.get(_model.config.model_type, "distilbert-base-uncased")
            _tok = AutoTokenizer.from_pretrained(_base)
        # DistilBERT / RoBERTa forward() don't accept token_type_ids, but a BERT
        # tokenizer emits them and the pipeline forwards them straight into the
        # model -> TypeError. Drop that input for models that don't use it.
        if _model.config.model_type in ("distilbert", "roberta"):
            _tok.model_input_names = [n for n in _tok.model_input_names if n != "token_type_ids"]
        MODEL = pipeline("text-classification", model=_model, tokenizer=_tok, top_k=None)
        # A char-level model was trained on space-separated characters; we must
        # feed it the same way at inference (flag saved in the model config).
        MODEL_CHAR_LEVEL = bool(getattr(_model.config, "char_level", False))
except Exception as exc:  # The heuristic path is intentionally always available.
    MODEL_ERROR = str(exc)
    MODEL = None


BRAND_CSS = """
/* ── Layout ─────────────────────────────────────────────────────────────── */
.gradio-container {
  max-width: 1180px !important;
  padding-left: clamp(12px, 4vw, 32px) !important;
  padding-right: clamp(12px, 4vw, 32px) !important;
}

/* ── Hero ────────────────────────────────────────────────────────────────── */
#hero {
  border-radius: 22px;
  padding: clamp(18px, 4vw, 28px);
  background: linear-gradient(135deg, rgba(25,33,58,.95), rgba(63,45,83,.92));
  color: white;
}
#hero h1 {
  font-size: clamp(1.6rem, 5vw, 2.4rem);
  margin-bottom: .25rem;
  /* Explicit white so high-contrast mode still reads it */
  color: #ffffff;
}
#hero p {
  font-size: clamp(.95rem, 2.5vw, 1.05rem);
  opacity: .94;
  color: #ffffff;
}

/* ── Educational boundary warning ──────────────────────────────────────── */
.warning-box {
  border-left: 5px solid #d97706;
  background: #fff7ed;
  color: #432818;
  padding: 14px 16px;
  border-radius: 12px;
  /* WCAG 1.4.3: contrast ratio ≥ 4.5:1 – #432818 on #fff7ed passes AA */
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.mode-card {
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  padding: 14px;
}

/* ── Buttons: minimum 44×44 px touch target (WCAG 2.5.5) ────────────────── */
button, .gr-button {
  min-height: 44px !important;
  min-width: 44px !important;
  font-size: clamp(.9rem, 2vw, 1rem) !important;
}

/* Primary CTA — ensure visible focus ring for keyboard navigation */
button:focus-visible, .gr-button:focus-visible {
  outline: 3px solid #2563eb !important;
  outline-offset: 3px !important;
}

/* ── Text inputs: readable size + sufficient contrast ────────────────────── */
textarea, input[type="text"] {
  font-size: clamp(.9rem, 2vw, 1rem) !important;
  line-height: 1.6 !important;
}

/* ── Tables in Markdown outputs: horizontal scroll on narrow screens ─────── */
.prose table, .md-block table {
  display: block;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  max-width: 100%;
}

/* ── Mobile: stack columns on small screens ─────────────────────────────── */
@media (max-width: 640px) {
  .gr-row {
    flex-direction: column !important;
  }
  .gr-row > * {
    width: 100% !important;
    max-width: 100% !important;
  }
}

/* ── Reduced-motion: disable decorative gradients ───────────────────────── */
@media (prefers-reduced-motion: reduce) {
  #hero { background: #1a2140; }
}

/* ── Dark mode ───────────────────────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
  .gradio-container {
    background: #0f1117 !important;
    color: #e8eaf0 !important;
  }
  .mode-card {
    border-color: #2d3148;
    background: #161b2e;
  }
  .warning-box {
    background: #2d1f0a;
    color: #f5d9a8;
    border-left-color: #d97706;
  }
  textarea, input[type="text"], .gr-input {
    background: #1a1f35 !important;
    color: #e8eaf0 !important;
    border-color: #2d3148 !important;
  }
  /* Keep markdown output text readable */
  .prose, .md-block {
    color: #e8eaf0 !important;
  }
}
"""


EXAMPLES = [
    ["WKLV LV D FODVVLFDO FDHVDU FLSKHU GHPR IRU FLSKHU GHWHFWLYH DL"],
    ["GSV XLWV RH ZOO BLFIH GSV VEVIVHG RMP"],
    ["LXFOPVEFRNHR"],
    ["TEITELHDVLSNHDTISEIIEA"],
    ["EOACT IPTRH IIEEN HSGES SOSCR REMEN AERTC OEFNT TYIHE THCMC"],
    ["THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY"],
]


def random_example() -> str:
    """Pick a random ciphertext example for the Detect tab."""
    return random.choice(EXAMPLES)[0]


def solve_substitution(ciphertext: str, iterations: int, restarts: int) -> tuple[str, str]:
    """Hill-climbing solver for monoalphabetic substitution. Educational only."""
    letters = clean_letters(ciphertext)
    if len(letters) < 30:
        return (
            "_Need at least ~30 letters of ciphertext for the climber to find a useful gradient._",
            "",
        )
    plaintext, key, score = hill_climb_substitution(
        ciphertext, iterations=int(iterations), restarts=int(restarts)
    )
    rletters = clean_letters(plaintext)
    chi = chi_squared_for_english(rletters)
    ws = word_score(plaintext)

    quality = ""
    if ws >= 3 and chi < 200:
        quality = "✅ **Looks like English.** Multiple word matches and reasonable letter frequencies."
    elif ws >= 1:
        quality = "🟡 **Partial match.** A few English words — try more iterations or another restart."
    else:
        quality = "❌ **Did not converge.** The sample may be too short, not English, or genuinely a different cipher family."

    key_table = "| Cipher | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z |\n"
    key_table += "|---|" + "|".join([f" {c} " for c in ALPHABET_LIST]) + "|\n"
    key_table += "| Plain |" + "|".join([f" **{c}** " for c in key]) + "|"

    output = (
        f"### Hill-climbing result\n"
        f"```\n{plaintext}\n```\n\n"
        f"{quality}\n\n"
        f"- Mean bigram log-prob: **{score:.3f}** (English ≈ −2.5 to −3.5)\n"
        f"- Word matches: **{ws}**\n"
        f"- Chi-squared vs English: **{chi:.1f}**\n\n"
        f"### Recovered key\n{key_table}\n"
    )
    note = (
        "_Hill climbing on English bigram log-probabilities, seeded from observed "
        "letter-frequency rank. Educational only: real cryptanalytic substitution "
        "solvers use richer n-gram models and smarter search. This converges on "
        "monoalphabetic substitutions of ≳120 letters of English; it will not solve "
        "polyalphabetic, transposition, or non-English ciphers — by design._"
    )
    return output, note


ALPHABET_LIST = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def transformer_predict(text: str) -> ModelPrediction | None:
    if MODEL is None:
        return None
    try:
        # Cap RAW characters before spacing so the char-level sequence (~1
        # token per character) stays under the model's 512-token limit. The
        # pipeline's own truncation=True is unreliable across transformers
        # versions, so we bound the input length ourselves.
        if MODEL_CHAR_LEVEL:
            model_input = char_tokenize_text(text[:500])
        else:
            model_input = text[:512]
        result = MODEL(model_input, truncation=True)
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]
        scores: dict[str, float] = {}
        for row in result:
            label = str(row["label"]).lower().replace("label_", "")
            scores[label] = float(row["score"])
        if not scores:
            return None
        label = max(scores, key=scores.get)
        # Calibrate: a raw softmax score is over-confident; map it to the model's
        # measured accuracy at that score (scripts/calibrate_transformer.py).
        confidence = calibrate_transformer_confidence(scores[label])
        return ModelPrediction(label=label, confidence=confidence, scores=scores, source="transformer")
    except Exception as exc:
        global TRANSFORMER_LAST_ERROR
        TRANSFORMER_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return None


# Cipher families where the heuristic is near-perfect (format-distinct: morse,
# polybius, tap code, digit codes, symbol ciphers). It should win there.
_HEURISTIC_STRONG_FAMILIES = {"code_format", "plain"}


# Quadgram log-prob per letter: real English ≈ -6.5, ciphertext ≈ -7.7 to -8.8.
# This threshold reliably answers "did the decode recover English?" — far more
# robust than word matching, which misses uncommon vocabulary.
_ENGLISH_QG = -7.2


def _looks_english(s: str) -> bool:
    letters = clean_letters(s)
    return len(letters) >= 20 and english_quadgram_score(letters) > _ENGLISH_QG


def _verified_decode(text: str) -> ModelPrediction | None:
    """If a simple attack actually BREAKS the text to clear English, we *know*
    the cipher — a recovered plaintext beats any classifier guess. Covers the
    brute-forceable ciphers (Caesar, ROT-13, Affine, Atbash) and auto-solvable
    Vigenère, exactly where the statistical classifiers confuse the look-alike
    shift ciphers. "Is it English?" is judged by quadgram score, not word lists.
    """
    if len(clean_letters(text)) < 25:
        return None
    if _looks_english(text):  # the input itself is English -> plaintext, not a cipher
        return None

    def _pred(label: str) -> ModelPrediction:
        return ModelPrediction(label, 0.92, {label: 0.92}, "verified-decode")

    # Caesar / ROT-13 — a non-zero brute-force shift recovers English.
    cc = best_caesar_candidates(text, 1)
    if cc and cc[0][0] != 0 and _looks_english(cc[0][3]):
        return _pred("rot13" if cc[0][0] == 13 else "caesar")
    # Atbash BEFORE affine (Atbash is the affine key a=25, b=25).
    if _looks_english(atbash(text)):
        return _pred("atbash")
    # Affine, excluding a=1 (Caesar) and a=25 (Atbash).
    ac = best_affine_candidates(text, 1)
    if ac and ac[0][0] not in (1, 25) and _looks_english(ac[0][4]):
        return _pred("affine")
    # Vigenère — Kasiski/Friedman auto-solve recovers English with a real key.
    vs = vigenere_auto_solve(text)
    if vs and len(vs[0][0]) >= 2 and _looks_english(vs[0][1]):
        return _pred("vigenere")
    return None


def combined_prediction(text: str) -> ModelPrediction:
    """Ensemble the transparent heuristic and the Transformer.

    1. If a simple attack actually decrypts the text to English, trust that —
       a broken cipher is a certainty, not a guess.
    2. The heuristic wins on format-distinct ciphers (~100%: Morse, Polybius,
       tap code, numeric codes, pigpen…).
    3. Otherwise take whichever predictor is more confident. Both confidences
       are calibrated (≈ probability correct), so this is an honest comparison —
       the model generalises better on the substitution / machine families.
    """
    verified = _verified_decode(text)
    if verified is not None:
        return verified
    heur = heuristic_classify(text)
    ml = transformer_predict(text)
    if ml is None:
        return heur
    if label_family(heur.label) in _HEURISTIC_STRONG_FAMILIES and heur.confidence >= 0.60:
        return heur
    return heur if heur.confidence >= ml.confidence else ml


def detective_mode(ciphertext: str) -> tuple[str, str]:
    if not clean_letters(ciphertext):
        return "Paste a classical ciphertext sample to begin.", ""
    pred = combined_prediction(ciphertext)
    explanation = build_explanation(ciphertext, pred)
    all_scores = sorted(pred.scores.items(), key=lambda kv: kv[1], reverse=True)
    top_scores = all_scores[:10]
    score_lines = ["| Label | Score |", "|---|---:|"]
    for label, score in top_scores:
        score_lines.append(f"| `{label}` | {score:.1%} |")
    if len(all_scores) > 10:
        score_lines.append(f"| _(+{len(all_scores) - 10} more)_ | … |")
    return explanation, "\n".join(score_lines)


def explain_only(ciphertext: str) -> str:
    if not clean_letters(ciphertext):
        return "Paste text first."
    ev = analyze_evidence(ciphertext)
    lines = [
        "## Evidence Notebook",
        f"- Letters analyzed: **{ev.letters}**",
        f"- Unique A–Z letters: **{ev.unique_letters}** / 26",
        f"- Index of coincidence: **{ev.index_of_coincidence}** (English ≈ 0.067, random ≈ 0.038)",
        f"- Shannon entropy: **{ev.entropy}** bits/letter",
        f"- English chi-squared: **{ev.chi_squared}** (lower = more English-like)",
        f"- Friedman key-length estimate: **{ev.friedman_key_length or '—'}**",
        f"- Transposition signal: **{ev.transposition_signal}** | bigram support: **{ev.bigram_support}**",
        "",
        "### Top bigrams",
        ", ".join([f"`{bg}` ({n})" for bg, n in ev.top_bigrams]) or "Not enough text.",
        "",
        "### Top trigrams",
        ", ".join([f"`{tg}` ({n})" for tg, n in ev.top_trigrams]) or "Not enough text.",
        "",
        "### Kasiski candidate key lengths",
        ", ".join(f"`{k}` (×{n})" for k, n in ev.kasiski_key_lengths) or "Not enough repeats.",
        "",
        "### Best Caesar / ROT candidates",
        "| Shift | Word clues | Chi² | Preview |",
        "|---:|---:|---:|---|",
    ]
    for shift, chi, score, decoded in ev.caesar_candidates[:3]:
        lines.append(f"| {shift} | {score} | {chi:.1f} | `{decoded[:60]}` |")
    if ev.affine_candidates:
        lines += [
            "",
            "### Best Affine candidates",
            "| a | b | Word clues | Chi² | Preview |",
            "|---:|---:|---:|---:|---|",
        ]
        for a, b, chi, score, decoded in ev.affine_candidates[:3]:
            lines.append(f"| {a} | {b} | {score} | {chi:.1f} | `{decoded[:55]}` |")
    lines += [
        "",
        "### Atbash reversal preview",
        f"`{ev.atbash_plaintext[:80]}`",
        "",
        "### Human reasoning",
    ]
    lines.extend(
        [f"- {note}" for note in ev.notes]
        or ["- The sample does not provide a strong single clue. Try a longer ciphertext."]
    )
    return "\n".join(lines)


DECODE_METHODS = [
    "auto-best-caesar",
    "auto-best-affine",
    "auto-vigenere",
    "auto-beaufort",
    "auto-columnar",
    "auto-rail-fence",
    "caesar_rot",
    "atbash",
    "vigenere",
    "beaufort",
    "affine",
    "columnar",
]


def try_decode(ciphertext: str, method: str, key_text: str) -> tuple[str, str]:
    """Attempt a manual decode and score the result."""
    letters = clean_letters(ciphertext)
    if not letters:
        return "Paste a ciphertext first.", ""

    result = ""
    note = ""

    if method == "auto-best-caesar":
        cands = best_caesar_candidates(ciphertext, top_n=5)
        lines = ["### Auto best-Caesar results", "| Shift | Word clues | Chi² | Plaintext |", "|---:|---:|---:|---|"]
        for shift, chi, score, decoded in cands:
            lines.append(f"| {shift} | {score} | {chi:.1f} | `{decoded[:70]}` |")
        return "\n".join(lines), "_Brute-forced all 26 Caesar shifts, ranked by English-word matches then chi-squared._"

    if method == "auto-best-affine":
        cands = best_affine_candidates(ciphertext, top_n=5)
        lines = ["### Auto best-Affine results", "| a | b | Word clues | Chi² | Plaintext |", "|---:|---:|---:|---:|---|"]
        for a, b, chi, score, decoded in cands:
            lines.append(f"| {a} | {b} | {score} | {chi:.1f} | `{decoded[:60]}` |")
        return "\n".join(lines), "_Brute-forced all 312 valid Affine keys (a coprime to 26), ranked by English-word matches._"

    if method == "auto-vigenere":
        cands = vigenere_auto_solve(ciphertext, max_key_len=15, top_n=5)
        if not cands:
            return "_Need at least 20 letters for Vigenère auto-solve._", ""
        lines = ["### Auto Vigenère results", "| Key | Word clues | Chi² | Plaintext preview |", "|---|---:|---:|---|"]
        for key, plaintext, chi, ws in cands:
            lines.append(f"| `{key}` | {ws} | {chi:.1f} | `{clean_letters(plaintext)[:65]}` |")
        return "\n".join(lines), "_Kasiski + Friedman key-length estimation, then per-column Caesar on each key position._"

    if method == "auto-rail-fence":
        cands = best_rail_fence_candidates(ciphertext, max_rails=15)
        lines = ["### Auto rail-fence results", "| Rails | Word clues | Chi² | Plaintext |", "|---:|---:|---:|---|"]
        for rails, decoded, chi, ws in cands:
            lines.append(f"| {rails} | {ws} | {chi:.1f} | `{decoded[:70]}` |")
        return "\n".join(lines), "_Brute-forced rail counts 2–15, ranked by English-word matches then chi-squared._"

    if method == "auto-beaufort":
        cands = beaufort_auto_solve(ciphertext, max_key_len=15, top_n=5)
        if not cands:
            return "_Need at least 20 letters for Beaufort auto-solve._", ""
        lines = ["### Auto Beaufort results", "| Key | Word clues | Chi² | Plaintext preview |", "|---|---:|---:|---|"]
        for key, plaintext, chi, ws in cands:
            lines.append(f"| `{key}` | {ws} | {chi:.1f} | `{clean_letters(plaintext)[:65]}` |")
        return "\n".join(lines), "_Kasiski + Friedman key-length estimation, then per-column Beaufort (reciprocal Vigenère) on each key position._"

    if method == "auto-columnar":
        cands = columnar_auto_solve(ciphertext, max_cols=8, top_n=5)
        if not cands:
            return "_Need at least 12 letters for columnar auto-solve._", ""
        lines = ["### Auto columnar results", "| Column order | Word clues | Quadgram | Plaintext preview |", "|---|---:|---:|---|"]
        for order, plaintext, qg, ws in cands:
            lines.append(f"| `{order}` | {ws} | {qg} | `{clean_letters(plaintext)[:60]}` |")
        return "\n".join(lines), "_For each column count 2–8, hill-climbs the column ordering to maximise English quadgram score (the key letters themselves are irrelevant — only their sort order matters)._"

    if method == "atbash":
        result = atbash(ciphertext)
        note = "Atbash is its own inverse — applied once."

    elif method == "caesar_rot":
        key_clean = key_text.strip()
        if key_clean.isdigit():
            shift = int(key_clean) % 26
        elif len(key_clean) == 1 and key_clean.isalpha():
            shift = ord(key_clean.upper()) - ord("A")
        else:
            return "_Key must be a number 0–25 or a single letter A–Z for Caesar._", ""
        result = caesar_shift(ciphertext, shift)
        note = f"Caesar shift {shift} applied."

    elif method == "vigenere":
        key_clean = clean_letters(key_text)
        if not key_clean:
            return "_Key must contain at least one A–Z letter for Vigenère._", ""
        try:
            result = vigenere_decrypt(ciphertext, key_clean)
            note = f"Vigenère decrypted with key `{key_clean}`."
        except ValueError as exc:
            return f"_Error: {exc}_", ""

    elif method == "beaufort":
        key_clean = clean_letters(key_text)
        if not key_clean:
            return "_Key must contain at least one A–Z letter for Beaufort._", ""
        try:
            result = beaufort_decrypt(ciphertext, key_clean)
            note = f"Beaufort decrypted with key `{key_clean}` (reciprocal cipher — same op encrypts and decrypts)."
        except ValueError as exc:
            return f"_Error: {exc}_", ""

    elif method == "affine":
        parts = [p.strip() for p in key_text.replace(",", " ").split()]
        if len(parts) != 2 or not all(p.lstrip("-").isdigit() for p in parts):
            return "_Affine key must be two integers: `a b` (e.g. `5 8`). a must be coprime with 26._", ""
        a, b = int(parts[0]), int(parts[1])
        try:
            result = affine_decrypt(ciphertext, a, b)
            note = f"Affine decrypted with a={a}, b={b}."
        except ValueError as exc:
            return f"_Error: {exc}_", ""

    elif method == "columnar":
        key_clean = clean_letters(key_text)
        if not key_clean:
            return "_Key must contain at least one A–Z letter for columnar transposition._", ""
        result = columnar_transposition_decrypt(ciphertext, key_clean)
        note = f"Columnar transposition decrypted with key `{key_clean}`."

    rletters = clean_letters(result)
    chi = chi_squared_for_english(rletters) if rletters else float("inf")
    ws = word_score(result)
    ent = shannon_entropy(rletters) if rletters else 0.0

    quality = "### Plaintext quality check"
    if ws >= 3 and chi < 100:
        quality += "\n✅ **Looks like English.** Multiple word matches and low chi-squared — this may be correct."
    elif ws >= 1:
        quality += "\n🟡 **Partial match.** Some English words found — check manually."
    else:
        quality += "\n❌ **Does not look like English.** No word matches found — try a different key."

    quality += f"\n- Word matches: **{ws}**\n- Chi-squared: **{chi:.1f}**\n- Entropy: **{ent:.3f}** bits/letter"

    output = f"### Decoded output\n```\n{result}\n```\n\n{quality}"
    return output, note


def compare_modes(ciphertext: str) -> tuple[str, str, str]:
    """Side-by-side: heuristic vs Transformer, plus an agreement summary."""
    if not clean_letters(ciphertext):
        return "Paste text first.", "Paste text first.", ""
    heur = heuristic_classify(ciphertext)
    ml = transformer_predict(ciphertext)

    def _table(p: ModelPrediction) -> str:
        all_s = sorted(p.scores.items(), key=lambda kv: kv[1], reverse=True)
        rows = ["| Label | Score |", "|---|---:|"]
        for label, score in all_s[:10]:
            rows.append(f"| `{label}` | {score:.1%} |")
        if len(all_s) > 10:
            rows.append(f"| _(+{len(all_s) - 10} more)_ | … |")
        return f"**Top: `{p.label}`** ({p.confidence:.1%})\n\n" + "\n".join(rows)

    heur_md = _table(heur)
    if ml is None:
        ml_md = (
            "_Transformer model not loaded._\n\n"
            "Set `CIPHER_MODEL_ID` to a Hugging Face model repo (e.g. "
            "`systemslibrarian/cipher-detective-classifier`) or train one locally with "
            "`scripts/train_transformer.py`."
        )
        agreement = "_Comparison unavailable — only the heuristic baseline is active._"
    else:
        ml_md = _table(ml)
        if ml.label == heur.label:
            agreement = (
                f"### ✅ Agreement\nBoth methods predict `{heur.label}`. "
                f"Combined confidence is reasonable when both agree."
            )
        else:
            agreement = (
                f"### ⚖️ Disagreement\n"
                f"- Heuristic: `{heur.label}` ({heur.confidence:.1%})\n"
                f"- Transformer: `{ml.label}` ({ml.confidence:.1%})\n\n"
                "Disagreements are *interesting*, not failures. Inspect the Evidence "
                "Notebook to decide which signal you trust more."
            )
    return heur_md, ml_md, agreement



def make_challenge(cipher_name: str, difficulty: str) -> tuple[str, str]:
    plaintexts = [
        "THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY",
        "CLASSICAL CIPHERS TEACH WHY MODERN SECURITY MATTERS",
        "THE DETECTIVE STUDIES PATTERNS BEFORE MAKING CLAIMS",
        "FREQUENCY ANALYSIS CAN REVEAL WEAK CIPHERS",
        "GOOD EDUCATIONAL TOOLS EXPLAIN THEIR LIMITS",
        "EVERY SYSTEM NEEDS AN HONEST THREAT MODEL",
    ]
    plain = random.choice(plaintexts)
    # Normalise legacy aliases to the canonical labels used by the trained
    # model and heuristic, so Challenge answers match Detect Mode output.
    _ALIAS = {"caesar_rot": "caesar", "columnar": "columnar_transposition",
              "substitution": "monoalphabetic"}
    label = _ALIAS.get(cipher_name, cipher_name)
    if label == "random":
        label = random.choice(
            ["caesar", "atbash", "vigenere", "rail_fence",
             "columnar_transposition", "affine", "monoalphabetic"]
        )
    if label == "caesar":
        shift = random.choice([3, 5, 7, 13, 19])
        return caesar_encrypt(plain, shift), f"Answer: Caesar / ROT shift {shift}. Plaintext: {plain}"
    if label == "atbash":
        return atbash(plain), f"Answer: Atbash. Plaintext: {plain}"
    if label == "vigenere":
        key = random.choice(["MUSEUM", "CIPHER", "LIBRARY", "SECURE"])
        return vigenere_encrypt(plain, key), f"Answer: Vigenère with key {key}. Plaintext: {plain}"
    if label == "rail_fence":
        rails = 3 if difficulty != "hard" else 4
        return rail_fence_encrypt(plain, rails), f"Answer: Rail Fence with {rails} rails. Plaintext: {plain}"
    if label == "columnar_transposition":
        key = random.choice(["MUSEUM", "LIBRARY", "PATTERN"])
        return columnar_transposition_encrypt(plain, key), f"Answer: Columnar transposition with key {key}. Plaintext: {plain}"
    if label == "affine":
        a, b = random.choice([(5, 8), (7, 3), (11, 6), (17, 9)])
        return affine_encrypt(plain, a, b), f"Answer: Affine cipher a={a}, b={b}. Plaintext: {plain}"
    if label == "monoalphabetic":
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        random.shuffle(alphabet)
        mapping = "".join(alphabet)
        return substitution_encrypt(plain, mapping), f"Answer: Monoalphabetic substitution with mapping {mapping}. Plaintext: {plain}"
    return plain, f"Answer: Plaintext. Plaintext: {plain}"


with gr.Blocks(title="Cipher Detective AI", css=BRAND_CSS) as demo:
    gr.HTML(
        """
        <header id="hero" role="banner">
          <h1>🕵️ Cipher Detective AI</h1>
          <p><strong>See the pattern. Test the hypothesis. Break the weak cipher. Respect the strong ones.</strong></p>
          <p>An educational Hugging Face Space combining transparent classical cryptanalysis with an optional Transformer classifier.</p>
        </header>
        """
    )
    gr.HTML(
        """
        <div class="warning-box" role="note" aria-label="Educational boundary notice">
          <strong>Educational boundary:</strong> this tool is for classical ciphers and cryptography education.
          It does not break modern encryption, recover passwords, bypass controls, or support unauthorized access.
        </div>
        """
    )

    with gr.Tab("Detect Mode"):
        with gr.Row():
            with gr.Column(scale=2):
                ciphertext = gr.Textbox(
                    label="Paste ciphertext",
                    lines=8,
                    placeholder="Example: WKLV LV D FDHVDU FLSKHU...",
                    info="Paste any classical ciphertext. Spaces and punctuation are tolerated.",
                )
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🕵️ Analyze like a detective",
                        variant="primary",
                        elem_id="analyze-btn",
                    )
                    random_btn = gr.Button(
                        "🎲 Load a random example",
                        variant="secondary",
                        elem_id="random-btn",
                    )
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[ciphertext],
                    label="Try an example",
                )
            with gr.Column(scale=1):
                scores = gr.Markdown(label="Confidence scores")
        report = gr.Markdown(label="Detective report")
        # Accept Enter key from the textbox and the click button.
        ciphertext.submit(detective_mode, inputs=[ciphertext], outputs=[report, scores])
        analyze_btn.click(detective_mode, inputs=[ciphertext], outputs=[report, scores])
        random_btn.click(random_example, inputs=None, outputs=[ciphertext])

    with gr.Tab("Explain Mode"):
        explain_input = gr.Textbox(
            label="Ciphertext",
            lines=7,
            placeholder="Paste any ciphertext to examine the evidence...",
        )
        explain_btn = gr.Button("Show evidence notebook", elem_id="explain-btn")
        explain_out = gr.Markdown(label="Evidence notebook")
        explain_input.submit(explain_only, inputs=[explain_input], outputs=[explain_out])
        explain_btn.click(explain_only, inputs=[explain_input], outputs=[explain_out])

    with gr.Tab("Challenge Mode"):
        gr.Markdown(
            "Generate an encrypted challenge and try to identify the cipher before revealing the answer.",
            elem_id="challenge-intro",
        )
        with gr.Row():
            cipher_choice = gr.Dropdown(
                ["random", "caesar", "atbash", "vigenere", "rail_fence",
                 "columnar_transposition", "affine", "monoalphabetic"],
                value="random",
                label="Challenge type",
            )
            difficulty = gr.Dropdown(
                ["easy", "medium", "hard"],
                value="medium",
                label="Difficulty",
            )
        challenge_btn = gr.Button("Generate challenge", variant="primary", elem_id="challenge-btn")
        challenge_text = gr.Textbox(
            label="Ciphertext challenge",
            lines=5,
            interactive=False,
        )
        answer = gr.Textbox(
            label="Reveal answer",
            lines=3,
            interactive=False,
        )
        challenge_btn.click(make_challenge, inputs=[cipher_choice, difficulty], outputs=[challenge_text, answer])

    with gr.Tab("Try Decode"):
        gr.Markdown(
            "Apply a specific cipher reversal with your guessed key, then see an automatic "
            "quality check. Use **auto-best-caesar** or **auto-best-affine** to brute-force "
            "without knowing the key."
        )
        gr.HTML("""
<div class="warning-box">
<strong>⚠️ What this tool can and cannot do</strong><br><br>
<strong>Can decode (math is fully reversible with the right key):</strong>
Caesar / ROT-N &nbsp;·&nbsp; ROT-13 &nbsp;·&nbsp; Atbash &nbsp;·&nbsp; Affine &nbsp;·&nbsp; Vigenère &nbsp;·&nbsp; Beaufort<br><br>
<strong>Cannot decode — the Detect tab identifies these, but cracking them requires:</strong><br>
&nbsp;• <em>Enigma / Lorenz / Typex / SIGABA / M209 / KL-7</em> — rotor settings &amp; plugboard state; keyspace is astronomical without cribs<br>
&nbsp;• <em>Playfair / Bifid / Four-square / Hill / Two-square</em> — key square brute-force is a 25! search space (needs a full hill-climber beyond scope here)<br>
&nbsp;• <em>Book cipher / Running key / Null cipher</em> — require the physical book or cover text<br>
&nbsp;• <em>Polybius / Tap / Morse / ADFGVX / Bacon</em> — format-conversion ciphers; deterministic once you know the grid, but not reversible from raw A–Z letters alone<br>
&nbsp;• <em>Navajo code / Copiale / Voynich</em> — language or symbol systems, not mathematical substitutions<br><br>
The classifier can tell you <em>which family</em> your ciphertext belongs to. Reversing it is a separate — often much harder — problem.
</div>
""")
        with gr.Row():
            with gr.Column(scale=2):
                decode_input = gr.Textbox(
                    label="Ciphertext",
                    lines=7,
                    placeholder="Paste ciphertext to attempt decoding...",
                )
            with gr.Column(scale=1):
                decode_method = gr.Dropdown(
                    DECODE_METHODS,
                    value="auto-best-caesar",
                    label="Decryption method",
                )
                decode_key = gr.Textbox(
                    label="Key (where required)",
                    placeholder="Caesar: 0–25 | Vigenère/Beaufort: word | Affine: a b | Columnar: keyword",
                    lines=1,
                    info="Leave blank for auto-* and Atbash methods. Caesar: number or letter. Affine: two integers e.g. '5 8'. Vigenère/Beaufort/Columnar: a word.",
                )
                decode_btn = gr.Button("Decode", variant="primary", elem_id="decode-btn")
        decode_out = gr.Markdown(label="Decoded result + quality check")
        decode_note = gr.Markdown(label="Method note")
        decode_btn.click(try_decode, inputs=[decode_input, decode_method, decode_key], outputs=[decode_out, decode_note])

    with gr.Tab("Compare Mode"):
        gr.Markdown(
            "Run the **transparent heuristic baseline** and the **Transformer classifier** "
            "on the same ciphertext. Disagreements are highlighted — they're often the most "
            "educational examples."
        )
        compare_input = gr.Textbox(
            label="Ciphertext",
            lines=7,
            placeholder="Paste ciphertext to compare methods...",
        )
        compare_btn = gr.Button("Compare methods", variant="primary", elem_id="compare-btn")
        with gr.Row():
            heur_out = gr.Markdown(label="Heuristic baseline")
            ml_out = gr.Markdown(label="Transformer classifier")
        agreement_out = gr.Markdown(label="Agreement summary")
        compare_input.submit(compare_modes, inputs=[compare_input], outputs=[heur_out, ml_out, agreement_out])
        compare_btn.click(compare_modes, inputs=[compare_input], outputs=[heur_out, ml_out, agreement_out])

    with gr.Tab("Solve Substitution"):
        gr.Markdown(
            "Hill-climb a monoalphabetic substitution cipher using English bigram "
            "log-probabilities. Best on **120+ letters** of English text. Short or "
            "non-English ciphertexts are *meant* to fail — that failure mode is part "
            "of the lesson on why classical ciphers can be broken at all."
        )
        with gr.Row():
            with gr.Column(scale=2):
                solve_input = gr.Textbox(
                    label="Substitution ciphertext",
                    lines=8,
                    placeholder="Paste a monoalphabetic substitution ciphertext (≥120 letters works best)...",
                )
            with gr.Column(scale=1):
                iters = gr.Slider(
                    minimum=500, maximum=10000, value=4000, step=500,
                    label="Iterations per restart",
                    info="Swaps attempted before giving up on a restart.",
                )
                restarts = gr.Slider(
                    minimum=1, maximum=8, value=3, step=1,
                    label="Random restarts",
                    info="Higher = more chances to escape local optima, slower.",
                )
                solve_btn = gr.Button(
                    "🧗 Hill-climb solve",
                    variant="primary",
                    elem_id="solve-btn",
                )
        solve_out = gr.Markdown(label="Recovered plaintext + key")
        solve_note = gr.Markdown(label="Method note")
        solve_btn.click(
            solve_substitution,
            inputs=[solve_input, iters, restarts],
            outputs=[solve_out, solve_note],
        )

    with gr.Tab("About / Model Status"):
        gr.Markdown(
            f"""
            ## Hugging Face-native design

            This project is designed as three artifacts:

            1. **Space** — this interactive exhibit.
            2. **Dataset** — `classical-cipher-corpus`, generated with `scripts/generate_dataset.py`.
            3. **Model** — `cipher-detective-classifier`, trained with `scripts/train_transformer.py`.

            ### Current model status

            | Property | Value |
            |---|---|
            | Transformer loaded | **{MODEL is not None}** |
            | Model source | `{os.getenv("CIPHER_MODEL_ID", "cipher_model")}` |
            | Heuristic fallback | **always available** |
            | Last loading error | `{MODEL_ERROR or "none"}` |

            ### Why this matters

            Classical ciphers are excellent teaching tools because they leak patterns.
            Modern cryptography is different: security depends on vetted primitives,
            protocols, implementation details, key management, metadata handling, and
            an honest threat model.
            """
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
