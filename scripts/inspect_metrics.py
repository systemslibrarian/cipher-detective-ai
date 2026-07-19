"""Summarize an evaluate_baseline.py report: worst classes, top confusions,
family-level metrics, and length buckets.

Usage:
    python scripts/inspect_metrics.py [reports/baseline_metrics.json]
"""
import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

path = sys.argv[1] if len(sys.argv) > 1 else "reports/baseline_metrics.json"
rep = json.load(open(path, encoding="utf-8"))
h = rep["heuristic"]
labels = rep["dataset"]["labels"]
print(f"rows: {rep['dataset']['size']}  classes: {len(labels)}")
print(f"fine-grained : accuracy={h['accuracy']:.4f}  macro_f1={h['macro_f1']:.4f}")
fam = h.get("family_level")
if fam:
    print(f"family-level : accuracy={fam['accuracy']:.4f}  macro_f1={fam['macro_f1']:.4f}")
    fr = fam["classification_report"]
    for f in fam["families"]:
        v = fr[f]
        print(f"    {f:20s} f1={v['f1-score']:.2f} p={v['precision']:.2f} "
              f"r={v['recall']:.2f} n={int(v['support'])}")

cr = h["classification_report"]
rows = [(k, v) for k, v in cr.items() if isinstance(v, dict) and k in labels]
rows.sort(key=lambda kv: kv[1]["f1-score"])
print("\nWORST 20 classes (f1 / precision / recall / support):")
for k, v in rows[:20]:
    print(f"  {k:28s} f1={v['f1-score']:.2f} p={v['precision']:.2f} r={v['recall']:.2f} n={int(v['support'])}")

cm = h["confusion_matrix"]
conf = []
for i, row in enumerate(cm):
    for j, n in enumerate(row):
        if i != j and n > 0:
            conf.append((n, labels[i], labels[j]))
conf.sort(reverse=True)
print("\nTOP 20 confusions (true -> predicted):")
for n, t, p in conf[:20]:
    print(f"  {n:5d}  {t:26s} -> {p}")

print("\nBy length bucket:")
for b, v in h["by_length"].items():
    print(f"  {b:12s} n={v['n']:6d} acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f}")
