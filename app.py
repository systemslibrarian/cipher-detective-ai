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
    affine_encrypt,
    analyze_evidence,
    atbash,
    build_explanation,
    caesar_encrypt,
    clean_letters,
    columnar_transposition_encrypt,
    heuristic_classify,
    rail_fence_encrypt,
    substitution_encrypt,
    vigenere_encrypt,
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
.gradio-container {max-width: 1180px !important}
#hero {
  border-radius: 22px;
  padding: 28px;
  background: linear-gradient(135deg, rgba(25,33,58,.95), rgba(63,45,83,.92));
  color: white;
}
#hero h1 {font-size: 2.4rem; margin-bottom: .25rem}
#hero p {font-size: 1.05rem; opacity: .94}
.warning-box {
  border-left: 5px solid #d97706;
  background: #fff7ed;
  color: #432818;
  padding: 14px 16px;
  border-radius: 12px;
}
.mode-card {
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  padding: 14px;
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
    score_lines = ["| Label | Score |", "|---|---:|"]
    for label, score in sorted(pred.scores.items(), key=lambda kv: kv[1], reverse=True):
        score_lines.append(f"| `{label}` | {score:.1%} |")
    return explanation, "\n".join(score_lines)


def explain_only(ciphertext: str) -> str:
    if not clean_letters(ciphertext):
        return "Paste text first."
    ev = analyze_evidence(ciphertext)
    lines = [
        "## Evidence Notebook",
        f"- Letters analyzed: **{ev.letters}**",
        f"- Index of coincidence: **{ev.index_of_coincidence}**",
        f"- Entropy: **{ev.entropy}** bits/letter",
        f"- Friedman key-length estimate: **{ev.friedman_key_length or '—'}**",
        f"- Transposition signal: **{ev.transposition_signal}** (bigram support **{ev.bigram_support}**)",
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
        "### Human reasoning",
    ]
    lines.extend(
        [f"- {note}" for note in ev.notes]
        or ["- The sample does not provide a strong single clue. Try a longer ciphertext."]
    )
    return "\n".join(lines)


def compare_modes(ciphertext: str) -> Tuple[str, str, str]:
    """Side-by-side: heuristic vs Transformer, plus an agreement summary."""
    if not clean_letters(ciphertext):
        return "Paste text first.", "Paste text first.", ""
    heur = heuristic_classify(ciphertext)
    ml = transformer_predict(ciphertext)

    def _table(p: ModelPrediction) -> str:
        rows = ["| Label | Score |", "|---|---:|"]
        for label, score in sorted(p.scores.items(), key=lambda kv: kv[1], reverse=True):
            rows.append(f"| `{label}` | {score:.1%} |")
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
        <div id="hero">
          <h1>🕵️‍♂️ Cipher Detective AI</h1>
          <p><strong>See the pattern. Test the hypothesis. Break the weak cipher. Respect the strong ones.</strong></p>
          <p>An educational Hugging Face Space combining transparent classical cryptanalysis with an optional Transformer classifier.</p>
        </div>
        """
    )
    gr.Markdown(
        """
        <div class="warning-box">
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
                )
                analyze_btn = gr.Button("Analyze like a detective", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=[ciphertext])
            with gr.Column(scale=1):
                scores = gr.Markdown(label="Scores")
        report = gr.Markdown(label="Detective report")
        analyze_btn.click(detective_mode, inputs=[ciphertext], outputs=[report, scores])

    with gr.Tab("Explain Mode"):
        explain_input = gr.Textbox(label="Ciphertext", lines=7)
        explain_btn = gr.Button("Show evidence notebook")
        explain_out = gr.Markdown()
        explain_btn.click(explain_only, inputs=[explain_input], outputs=[explain_out])

    with gr.Tab("Challenge Mode"):
        with gr.Row():
            cipher_choice = gr.Dropdown(
                ["random", "caesar_rot", "atbash", "vigenere", "rail_fence", "columnar", "affine", "substitution"],
                value="random",
                label="Challenge type",
            )
            difficulty = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Difficulty")
        challenge_btn = gr.Button("Generate challenge")
        challenge_text = gr.Textbox(label="Ciphertext challenge", lines=5)
        answer = gr.Textbox(label="Reveal answer", lines=3)
        challenge_btn.click(make_challenge, inputs=[cipher_choice, difficulty], outputs=[challenge_text, answer])

    with gr.Tab("Compare Mode"):
        gr.Markdown(
            "Run the **transparent heuristic baseline** and the **Transformer classifier** "
            "on the same ciphertext. Disagreements are highlighted — they're often the most "
            "educational examples."
        )
        compare_input = gr.Textbox(label="Ciphertext", lines=7)
        compare_btn = gr.Button("Compare methods", variant="primary")
        with gr.Row():
            heur_out = gr.Markdown(label="Heuristic baseline")
            ml_out = gr.Markdown(label="Transformer classifier")
        agreement_out = gr.Markdown()
        compare_btn.click(
            compare_modes,
            inputs=[compare_input],
            outputs=[heur_out, ml_out, agreement_out],
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

            - Transformer loaded: **{MODEL is not None}**
            - Model source: `{os.getenv("CIPHER_MODEL_ID", "cipher_model")}`
            - Fallback available: **true**
            - Last model loading error: `{MODEL_ERROR or "none"}`

            ### Why this matters

            Classical ciphers are excellent teaching tools because they leak patterns.
            Modern cryptography is different: security depends on vetted primitives,
            protocols, implementation details, key management, metadata handling, and
            an honest threat model.
            """
        )

if __name__ == "__main__":
    demo.launch()
