from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import heuristic_classify

def load_rows(path: str):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cipher_examples.jsonl")
    ap.add_argument("--out", default="reports/baseline_metrics.json")
    args = ap.parse_args()

    y_true, y_pred = [], []
    for row in load_rows(args.data):
        y_true.append(row["label"])
        y_pred.append(heuristic_classify(row["text"]).label)

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "note": "Heuristic baseline is intentionally transparent and imperfect. Use it as a comparison point for the Transformer model.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": metrics["accuracy"], "out": str(out)}, indent=2))

if __name__ == "__main__":
    main()
