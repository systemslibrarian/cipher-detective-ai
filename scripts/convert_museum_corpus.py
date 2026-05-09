"""
Convert the cipher-museum public corpus (all.jsonl) into the training format
expected by train_transformer.py, using all 82 cipher types as labels.

Usage:
    python scripts/convert_museum_corpus.py \
        --museum  /path/to/cipher-museum/public/corpus/all.jsonl \
        --out     data/cipher_examples.jsonl \
        --split   public          # 'public', 'blind', or 'all'
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--museum", required=True, help="Path to cipher-museum all.jsonl")
    ap.add_argument("--out", default="data/cipher_examples.jsonl")
    ap.add_argument(
        "--split",
        default="public",
        choices=["public", "blind", "all"],
        help="Which corpus split to include (default: public)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    rows_in = []
    with open(args.museum, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.split != "all" and r.get("split", "public") != args.split:
                continue
            rows_in.append(r)

    print(f"Loaded {len(rows_in)} rows from museum corpus (split={args.split})")

    out_rows = []
    skipped = 0
    for r in rows_in:
        ciphertext = r.get("ciphertext", "").strip()
        label = r.get("cipher_type", "").strip()
        if not ciphertext or not label:
            skipped += 1
            continue

        out_rows.append(
            {
                "id": r.get("id", ""),
                "text": ciphertext,
                "ciphertext": ciphertext,
                "plaintext": r.get("plaintext", ""),
                "label": label,
                "cipher": label,
                "cipher_family": r.get("cipher_family", ""),
                "key": r.get("key", {}),
                "difficulty": r.get("difficulty", ""),
                "language": r.get("language", "en"),
                "text_length": len(ciphertext.replace(" ", "")),
                "length": len(ciphertext),
                "source": "cipher_museum",
                "dataset_version": r.get("dataset_version", ""),
                "split": r.get("split", ""),
            }
        )

    if skipped:
        print(f"Skipped {skipped} rows with missing ciphertext or label")

    random.shuffle(out_rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")

    # Print label distribution
    from collections import Counter
    label_counts = Counter(r["label"] for r in out_rows)
    print(f"\nWrote {len(out_rows)} rows → {args.out}")
    print(f"Unique labels: {len(label_counts)}")
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {count:6d}  {label}")


if __name__ == "__main__":
    main()
