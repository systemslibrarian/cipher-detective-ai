"""Fit a confidence-calibration map for the Transformer classifier.

Runs the model over a sample of the validation split, collects
(raw_confidence, correct) pairs, fits a monotonic (isotonic) map, and writes
transformer_calibration_map.json. app.py loads it so the model's reported
confidence matches its real accuracy (the same treatment the heuristic gets).

Usage:
    python scripts/calibrate_transformer.py --n 3000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

# Reuse the isotonic + reliability helpers from the heuristic calibrator.
from calibrate_confidence import ece, isotonic_fit, reliability_bins  # noqa: E402

from core import char_tokenize_text  # noqa: E402

_BASE_TOKENIZERS = {"distilbert": "distilbert-base-uncased",
                    "bert": "bert-base-uncased", "roberta": "roberta-base"}


def load_model(model_id: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    vocab = model.get_input_embeddings().weight.shape[0]
    if max(tok("probe test", truncation=True, max_length=8)["input_ids"]) >= vocab:
        tok = AutoTokenizer.from_pretrained(
            _BASE_TOKENIZERS.get(model.config.model_type, "distilbert-base-uncased"))
    if model.config.model_type in ("distilbert", "roberta"):
        tok.model_input_names = [n for n in tok.model_input_names if n != "token_type_ids"]
    char_level = bool(getattr(model.config, "char_level", False))
    return pipeline("text-classification", model=model, tokenizer=tok, top_k=1), char_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/splits/val.jsonl")
    ap.add_argument("--model", default="systemslibrarian/cipher-detective-classifier")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--map", default="transformer_calibration_map.json")
    ap.add_argument("--out", default="reports/transformer_calibration.json")
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    random.seed(42)
    if args.n < len(rows):
        rows = random.sample(rows, args.n)

    pipe, char_level = load_model(args.model)
    pairs = []
    for i, r in enumerate(rows):
        text = char_tokenize_text(r["text"][:500]) if char_level else r["text"][:512]
        out = pipe(text, truncation=True)
        row = out[0][0] if isinstance(out[0], list) else out[0]
        conf = round(float(row["score"]), 4)
        pred = str(row["label"]).lower().replace("label_", "")
        pairs.append((conf, int(pred == r["label"])))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")

    bins = reliability_bins(pairs)
    fine_map = isotonic_fit(pairs)
    cal_pairs = []  # measure post-calibration ECE with the map we just fit
    xs = [x for x, _ in fine_map]
    ys = [y for _, y in fine_map]

    def _cal(c):
        if not xs:
            return c
        if c <= xs[0]:
            return ys[0]
        if c >= xs[-1]:
            return ys[-1]
        for j in range(len(xs) - 1):
            if xs[j] <= c <= xs[j + 1]:
                t = (c - xs[j]) / (xs[j + 1] - xs[j]) if xs[j + 1] > xs[j] else 0
                return ys[j] + t * (ys[j + 1] - ys[j])
        return c

    cal_pairs = [(_cal(c), y) for c, y in pairs]
    compact = {str(round(x, 4)): round(y, 4) for x, y in fine_map}
    Path(args.map).write_text(json.dumps({"fine_confidence_to_accuracy": compact}, indent=2), encoding="utf-8")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    acc = sum(y for _, y in pairs) / len(pairs)
    Path(args.out).write_text(json.dumps(
        {"n": len(pairs), "accuracy": round(acc, 4),
         "ece_raw": ece(bins), "ece_calibrated": ece(reliability_bins(cal_pairs))}, indent=2), encoding="utf-8")
    print(f"n={len(pairs)} accuracy={acc:.4f}")
    print(f"ECE raw={ece(bins)} -> calibrated={ece(reliability_bins(cal_pairs))}")
    print(f"wrote {args.map} ({len(compact)} breakpoints)")


if __name__ == "__main__":
    main()
