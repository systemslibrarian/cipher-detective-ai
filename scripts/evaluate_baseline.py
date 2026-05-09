from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import heuristic_classify


def load_rows(path: str):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def transformer_predictions(texts, model_id: str):
    """Optional transformer predictions. Returns ``None`` if unavailable."""
    try:
        from transformers import pipeline
    except Exception:  # transformers not installed
        return None
    try:
        pipe = pipeline("text-classification", model=model_id, tokenizer=model_id, top_k=1)
    except Exception as exc:  # model can't be loaded
        print(f"[evaluate_baseline] Transformer unavailable: {exc}")
        return None
    preds = []
    for t in texts:
        out = pipe(t[:512])
        # `top_k=1` returns a list-of-list; flatten.
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        preds.append(str(out[0]["label"]).lower().replace("label_", ""))
    return preds


def evaluate(y_true, y_pred, labels):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _length_bucket(n: int) -> str:
    if n < 50:
        return "xs (<50)"
    if n < 100:
        return "s (50-99)"
    if n < 200:
        return "m (100-199)"
    if n < 400:
        return "l (200-399)"
    return "xl (>=400)"


def bucketed_metrics(rows, y_true, y_pred, labels, key):
    """Group accuracy + macro-F1 by a row attribute (e.g. difficulty, length bucket)."""
    buckets: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if key == "length_bucket":
            bucket = _length_bucket(int(r.get("text_length") or r.get("length") or 0))
        else:
            bucket = str(r.get(key, "unknown"))
        buckets.setdefault(bucket, []).append(i)
    out = {}
    for bucket, idxs in sorted(buckets.items()):
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        out[bucket] = {
            "n": len(idxs),
            "accuracy": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate the heuristic baseline (and optionally a Transformer model).")
    ap.add_argument("--data", default="data/cipher_examples.jsonl")
    ap.add_argument("--out", default="reports/baseline_metrics.json")
    ap.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample this many rows (stratified per label) for a quick evaluation.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--model",
        default=os.getenv("CIPHER_MODEL_ID", ""),
        help="Optional Hugging Face model ID or local path. If set, results are reported alongside the heuristic.",
    )
    args = ap.parse_args()

    rows = list(load_rows(args.data))

    if args.sample and args.sample < len(rows):
        import random as _random
        _random.seed(args.seed)
        # Stratified sample: take proportional count from each label
        from collections import defaultdict as _dd
        by_label: dict = _dd(list)
        for r in rows:
            by_label[r["label"]].append(r)
        sampled: list = []
        per_label = max(1, args.sample // len(by_label))
        for lbl_rows in by_label.values():
            k = min(per_label, len(lbl_rows))
            sampled.extend(_random.sample(lbl_rows, k))
        rows = sampled
        print(f"Sampled {len(rows)} rows ({per_label} per label, {len(by_label)} labels)")

    texts = [r["text"] for r in rows]
    y_true = [r["label"] for r in rows]
    labels = sorted(set(y_true))

    y_pred_heur = [heuristic_classify(t).label for t in texts]
    # Map any "too_short" predictions to a neutral fallback so metrics stay well-defined.
    y_pred_heur = [p if p in labels else "plaintext" for p in y_pred_heur]

    heuristic_block = evaluate(y_true, y_pred_heur, labels)
    heuristic_block["by_difficulty"] = bucketed_metrics(rows, y_true, y_pred_heur, labels, "difficulty")
    heuristic_block["by_length"] = bucketed_metrics(rows, y_true, y_pred_heur, labels, "length_bucket")

    report = {
        "dataset": {
            "path": args.data,
            "size": len(rows),
            "labels": labels,
            "label_distribution": dict(Counter(y_true)),
        },
        "heuristic": heuristic_block,
        "note": (
            "Heuristic baseline is intentionally transparent and imperfect. "
            "Use it as a comparison point for the Transformer model. None of these "
            "metrics imply real-world cryptanalytic capability."
        ),
    }

    if args.model:
        ml_preds = transformer_predictions(texts, args.model)
        if ml_preds is not None:
            ml_preds = [p if p in labels else "plaintext" for p in ml_preds]
            ml_block = evaluate(y_true, ml_preds, labels)
            ml_block["by_difficulty"] = bucketed_metrics(rows, y_true, ml_preds, labels, "difficulty")
            ml_block["by_length"] = bucketed_metrics(rows, y_true, ml_preds, labels, "length_bucket")
            report["transformer"] = {"model_id": args.model, **ml_block}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {"out": str(out), "heuristic_accuracy": report["heuristic"]["accuracy"]}
    if "transformer" in report:
        summary["transformer_accuracy"] = report["transformer"]["accuracy"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
