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

import os
import random
from typing import Dict, Tuple

import gradio as gr

from core import (
    ModelPrediction,
    affine_decrypt,
    affine_encrypt,
    analyze_evidence,
    atbash,
    best_affine_candidates,
    best_caesar_candidates,
    build_explanation,
    caesar_encrypt,
    caesar_shift,
    chi_squared_for_english,
    clean_letters,
    columnar_transposition_encrypt,
    heuristic_classify,
    hill_climb_substitution,
    rail_fence_encrypt,
    shannon_entropy,
    substitution_encrypt,
    vigenere_decrypt,
    vigenere_encrypt,
    word_score,
)

MODEL = None
MODEL_LABELS = None
MODEL_ERROR = None

try:
    from transformers import pipeline

    model_id = os.getenv("CIPHER_MODEL_ID", "cipher_model")
    if os.path.isdir(model_id) or "/" in model_id:
        MODEL = pipeline("text-classification", model=model_id, tokenizer=model_id, top_k=None)
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


def solve_substitution(ciphertext: str, iterations: int, restarts: int) -> Tuple[str, str]:
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
        result = MODEL(text[:512])
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]
        scores: Dict[str, float] = {}
        for row in result:
            label = str(row["label"]).lower().replace("label_", "")
            scores[label] = float(row["score"])
        if not scores:
            return None
        label = max(scores, key=scores.get)
        return ModelPrediction(label=label, confidence=scores[label], scores=scores, source="transformer")
    except Exception:
        return None


def combined_prediction(text: str) -> ModelPrediction:
    ml = transformer_predict(text)
    if ml:
        return ml
    return heuristic_classify(text)


def detective_mode(ciphertext: str) -> Tuple[str, str]:
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


DECODE_METHODS = ["caesar_rot", "atbash", "vigenere", "affine", "auto-best-caesar", "auto-best-affine"]


def try_decode(ciphertext: str, method: str, key_text: str) -> Tuple[str, str]:
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


def compare_modes(ciphertext: str) -> Tuple[str, str, str]:
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



def make_challenge(cipher_name: str, difficulty: str) -> Tuple[str, str]:
    plaintexts = [
        "THE LIBRARY PRESERVES KNOWLEDGE FOR THE COMMUNITY",
        "CLASSICAL CIPHERS TEACH WHY MODERN SECURITY MATTERS",
        "THE DETECTIVE STUDIES PATTERNS BEFORE MAKING CLAIMS",
        "FREQUENCY ANALYSIS CAN REVEAL WEAK CIPHERS",
        "GOOD EDUCATIONAL TOOLS EXPLAIN THEIR LIMITS",
        "EVERY SYSTEM NEEDS AN HONEST THREAT MODEL",
    ]
    plain = random.choice(plaintexts)
    label = cipher_name
    if cipher_name == "random":
        label = random.choice(
            ["caesar_rot", "atbash", "vigenere", "rail_fence", "columnar", "affine", "substitution"]
        )
    if label == "caesar_rot":
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
    if label == "columnar":
        key = random.choice(["MUSEUM", "LIBRARY", "PATTERN"])
        return columnar_transposition_encrypt(plain, key), f"Answer: Columnar transposition with key {key}. Plaintext: {plain}"
    if label == "affine":
        a, b = random.choice([(5, 8), (7, 3), (11, 6), (17, 9)])
        return affine_encrypt(plain, a, b), f"Answer: Affine cipher a={a}, b={b}. Plaintext: {plain}"
    if label == "substitution":
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        random.shuffle(alphabet)
        mapping = "".join(alphabet)
        return substitution_encrypt(plain, mapping), f"Answer: Monoalphabetic substitution with mapping {mapping}. Plaintext: {plain}"
    return plain, f"Answer: Plaintext. Plaintext: {plain}"


with gr.Blocks(css=BRAND_CSS, title="Cipher Detective AI") as demo:
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
                scores = gr.Markdown(label="Confidence scores", show_label=True)
        report = gr.Markdown(label="Detective report", show_label=True)
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
        explain_out = gr.Markdown(label="Evidence notebook", show_label=True)
        explain_input.submit(explain_only, inputs=[explain_input], outputs=[explain_out])
        explain_btn.click(explain_only, inputs=[explain_input], outputs=[explain_out])

    with gr.Tab("Challenge Mode"):
        gr.Markdown(
            "Generate an encrypted challenge and try to identify the cipher before revealing the answer.",
            elem_id="challenge-intro",
        )
        with gr.Row():
            cipher_choice = gr.Dropdown(
                ["random", "caesar_rot", "atbash", "vigenere", "rail_fence", "columnar", "affine", "substitution"],
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
                    placeholder="Caesar: shift 0–25 or letter | Vigenère: word | Affine: a b",
                    lines=1,
                    info="Number for Caesar, word for Vigenère, two ints (a b) for Affine. Leave blank for auto / Atbash.",
                )
                decode_btn = gr.Button("Decode", variant="primary", elem_id="decode-btn")
        decode_out = gr.Markdown(label="Decoded result + quality check", show_label=True)
        decode_note = gr.Markdown(label="Method note", show_label=True)
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
            heur_out = gr.Markdown(label="Heuristic baseline", show_label=True)
            ml_out = gr.Markdown(label="Transformer classifier", show_label=True)
        agreement_out = gr.Markdown(label="Agreement summary", show_label=True)
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
        solve_out = gr.Markdown(label="Recovered plaintext + key", show_label=True)
        solve_note = gr.Markdown(label="Method note", show_label=True)
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
    demo.launch()
