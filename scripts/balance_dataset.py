"""
Balance and split the cipher dataset for Transformer training.

Takes the full cipher_examples.jsonl (90k+) and produces:
  - data/train.jsonl   — balanced training set (capped per class)
  - data/val.jsonl     — 10% stratified validation
  - data/test.jsonl    — held-out test (uses cipher_examples_blind.jsonl if present)

Usage:
    python scripts/balance_dataset.py
    python scripts/balance_dataset.py --max-per-class 1200 --val-frac 0.1
    python scripts/balance_dataset.py --out-dir data/splits --seed 99

Then train:
    python scripts/train_transformer.py \\
        --data data/splits/train.jsonl \\
        --test-data data/splits/val.jsonl \\
        --model roberta-base \\
        --epochs 10 --batch-size 32 --grad-accum 2 \\
        --focal-loss --push-to-hub \\
        --hub-model-id systemslibrarian/cipher-detective-classifier
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Balance and split the cipher dataset.")
    ap.add_argument(
        "--data", default="data/cipher_examples.jsonl",
        help="Full training corpus (default: data/cipher_examples.jsonl)",
    )
    ap.add_argument(
        "--blind", default="data/cipher_examples_blind.jsonl",
        help="Blind test corpus (default: data/cipher_examples_blind.jsonl). "
             "If absent, test split is carved from --data.",
    )
    ap.add_argument(
        "--out-dir", default="data/splits",
        help="Output directory for train/val/test JSONL files.",
    )
    ap.add_argument(
        "--max-per-class", type=int, default=1000,
        help="Cap majority classes at this many examples (default: 1000). "
             "Minority classes are kept in full; majority are downsampled.",
    )
    ap.add_argument(
        "--val-frac", type=float, default=0.10,
        help="Fraction of (balanced) training data to hold out as validation (default: 0.10).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {data_path}")

    rows = load_jsonl(data_path)
    print(f"Loaded {len(rows):,} examples from {data_path}")

    # Group by label
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    labels = sorted(by_label)
    print(f"Labels: {len(labels)}")

    # ------------------------------------------------------------------
    # Balance: cap majority, keep all minority
    # ------------------------------------------------------------------
    balanced: list[dict] = []
    cap = args.max_per_class
    for label in labels:
        pool = by_label[label]
        random.shuffle(pool)
        balanced.extend(pool[:cap])

    random.shuffle(balanced)
    print(
        f"Balanced corpus: {len(balanced):,} examples "
        f"(cap={cap}/class, min={min(len(by_label[l]) for l in labels)}, "
        f"max={max(len(by_label[l]) for l in labels)} before cap)"
    )

    # ------------------------------------------------------------------
    # Stratified train / val split
    # ------------------------------------------------------------------
    val_frac = args.val_frac
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    by_label_balanced: dict[str, list[dict]] = defaultdict(list)
    for r in balanced:
        by_label_balanced[r["label"]].append(r)

    for label in labels:
        pool = by_label_balanced[label]
        n_val = max(1, round(len(pool) * val_frac))
        val_rows.extend(pool[:n_val])
        train_rows.extend(pool[n_val:])

    random.shuffle(train_rows)
    random.shuffle(val_rows)

    # ------------------------------------------------------------------
    # Test split: use blind file if available, otherwise carve from train
    # ------------------------------------------------------------------
    blind_path = Path(args.blind)
    if blind_path.exists():
        test_rows = load_jsonl(blind_path)
        # Filter to labels present in training
        train_labels = set(r["label"] for r in train_rows)
        test_rows = [r for r in test_rows if r["label"] in train_labels]
        print(f"Blind test set: {len(test_rows):,} examples from {blind_path}")
    else:
        print(f"No blind file at {blind_path} — carving 10% from train as test.")
        test_rows = []
        by_label_train: dict[str, list[dict]] = defaultdict(list)
        for r in train_rows:
            by_label_train[r["label"]].append(r)
        new_train: list[dict] = []
        for label in labels:
            pool = by_label_train.get(label, [])
            n_test = max(1, round(len(pool) * 0.10))
            test_rows.extend(pool[:n_test])
            new_train.extend(pool[n_test:])
        train_rows = new_train
        random.shuffle(train_rows)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    print(f"\nWriting splits to {out_dir}/")
    write_jsonl(train_rows, out_dir / "train.jsonl")
    write_jsonl(val_rows, out_dir / "val.jsonl")
    write_jsonl(test_rows, out_dir / "test.jsonl")

    print(f"\nSummary:")
    print(f"  Train : {len(train_rows):,} examples")
    print(f"  Val   : {len(val_rows):,} examples")
    print(f"  Test  : {len(test_rows):,} examples")
    print(f"  Labels: {len(labels)}")

    # Show the 5 smallest classes after balancing
    by_label_train_final: dict[str, int] = defaultdict(int)
    for r in train_rows:
        by_label_train_final[r["label"]] += 1
    smallest = sorted(by_label_train_final.items(), key=lambda x: x[1])[:5]
    print(f"\n  5 smallest training classes after balancing:")
    for label, n in smallest:
        print(f"    {label:40s} {n}")

    print("\nNext step — train on HF Spaces (A10G GPU) or Colab:")
    print(
        "  python scripts/train_transformer.py \\\n"
        f"      --data {out_dir}/train.jsonl \\\n"
        f"      --test-data {out_dir}/val.jsonl \\\n"
        "      --model roberta-base \\\n"
        "      --epochs 10 --batch-size 32 --grad-accum 2 \\\n"
        "      --focal-loss --push-to-hub \\\n"
        "      --hub-model-id systemslibrarian/cipher-detective-classifier"
    )


if __name__ == "__main__":
    main()
