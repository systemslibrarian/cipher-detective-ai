"""Derive full English bigram (26x26) and trigram log10-probability tables from
the plaintext fields of the cipher corpus, and emit them as compact Python
literals for embedding in core.py.

Usage:
    python scripts/build_ngram_tables.py > ngram_tables_generated.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[1]

texts: list[str] = []
seen = set()
with open(REPO / "data" / "cipher_examples.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pt = r.get("plaintext") or ""
        if isinstance(pt, str) and len(pt) > 20 and pt not in seen:
            seen.add(pt)
            texts.append(pt)

corpus = re.sub(r"[^A-Z]", "", " ".join(texts).upper())
print(f"# Derived from {len(texts):,} unique plaintexts, {len(corpus):,} letters",
      file=sys.stderr)

bi = Counter(corpus[i:i + 2] for i in range(len(corpus) - 1))
tri = Counter(corpus[i:i + 3] for i in range(len(corpus) - 2))
quad = Counter(corpus[i:i + 4] for i in range(len(corpus) - 3))

total_bi = sum(bi.values())
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Full 26x26 bigram table with add-one smoothing (no gaps -> smooth gradient).
print("_BIGRAM_LOG_PROB: dict[str, float] = {")
row_items = []
for a in ALPHA:
    for b in ALPHA:
        p = (bi.get(a + b, 0) + 1) / (total_bi + 676)
        row_items.append(f'"{a}{b}":{math.log10(p):.2f}')
for i in range(0, len(row_items), 8):
    print("    " + ", ".join(row_items[i:i + 8]) + ",")
print("}")

def emit(name: str, counter: Counter, top_n: int, per_line: int = 8) -> None:
    total = sum(counter.values())
    top = counter.most_common(top_n)
    print(f"{name}: dict[str, float] = {{")
    items = [f'"{t}":{math.log10(c / total):.2f}' for t, c in top]
    for i in range(0, len(items), per_line):
        print("    " + ", ".join(items[i:i + per_line]) + ",")
    print("}")
    print(f"# {name} coverage of corpus: {sum(c for _, c in top) / total:.1%}",
          file=sys.stderr)


emit("_TRIGRAM_LOG_PROB", tri, 3000)
emit("_QUADGRAM_LOG_PROB", quad, 5000)
