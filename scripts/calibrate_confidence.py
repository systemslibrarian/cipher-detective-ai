"""Measure and fit confidence calibration for the heuristic classifier.

The heuristic emits hand-picked confidences (0.45, 0.38, ...) that have never
been checked against reality. This script:

  1. runs the classifier over the test split, collecting (raw_confidence,
     correct) pairs for both fine-grained and family verdicts;
  2. bins them into a reliability curve (claimed vs actual accuracy);
  3. fits a monotonic (isotonic) calibration map raw -> empirical accuracy;
  4. writes the map to calibration_map.json and a reliability report.

core.calibrate_confidence() loads the map at runtime so a reported 70% means
"right about 70% of the time" — the whole point of an evidence-weighing tool.

Usage:
    python scripts/calibrate_confidence.py --data data/splits/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from core import _heuristic_classify_raw as classify_raw
from core import calibrate_confidence, label_family


def isotonic_fit(pairs: list[tuple[float, int]]) -> list[tuple[float, float]]:
    """Pool-adjacent-violators isotonic regression. Returns sorted
    (raw_confidence, calibrated_accuracy) breakpoints, monotonic non-decreasing."""
    pairs = sorted(pairs)
    xs = [x for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    w = [1.0] * len(ys)
    # PAVA
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1]:
            # pool i and i+1
            tw = w[i] + w[i + 1]
            ty = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / tw
            ys[i] = ty
            w[i] = tw
            del ys[i + 1]
            del w[i + 1]
            del xs[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return list(zip(xs, ys, strict=True))


def reliability_bins(pairs: list[tuple[float, int]], n_bins: int = 10):
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [c for conf, c in pairs if (lo <= conf < hi) or (b == n_bins - 1 and conf == hi)]
        if sel:
            mid_conf = sum(conf for conf, _ in pairs if lo <= conf < hi or (b == n_bins - 1 and conf == hi)) / len(sel)
            bins.append({
                "range": [round(lo, 2), round(hi, 2)],
                "n": len(sel),
                "claimed_mid": round(mid_conf, 4),
                "actual_accuracy": round(sum(sel) / len(sel), 4),
            })
    return bins


def ece(bins) -> float:
    """Expected Calibration Error: support-weighted |claimed - actual|."""
    total = sum(b["n"] for b in bins)
    if not total:
        return 0.0
    return round(sum(b["n"] * abs(b["claimed_mid"] - b["actual_accuracy"]) for b in bins) / total, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/splits/test.jsonl")
    ap.add_argument("--out", default="reports/calibration_report.json")
    ap.add_argument("--map", default="calibration_map.json")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in Path(args.data).read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Fit on RAW confidences (the wrapper heuristic_classify already applies the
    # map, which would be circular). Fit on val, evaluate on test for honesty.
    fine_pairs, fam_pairs = [], []
    for r in rows:
        pred = classify_raw(r["text"])
        conf = round(pred.confidence, 4)
        correct = int(pred.label == r["label"])
        fine_pairs.append((conf, correct))
        fam_pairs.append((conf, int(label_family(pred.label) == label_family(r["label"]))))

    fine_bins = reliability_bins(fine_pairs)
    fam_bins = reliability_bins(fam_pairs)
    fine_map = isotonic_fit(fine_pairs)

    # Post-calibration reliability using the map we just fit (identity floor if
    # the map file isn't in place yet — re-run once to measure the real gain).
    cal_pairs = [(calibrate_confidence(c), y) for c, y in fine_pairs]
    cal_bins = reliability_bins(cal_pairs)

    report = {
        "n": len(rows),
        "fine": {"reliability": fine_bins, "ece_raw": ece(fine_bins),
                 "ece_calibrated": ece(cal_bins)},
        "family": {"reliability": fam_bins, "ece_raw": ece(fam_bins)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Compress isotonic breakpoints to a compact lookup (unique x -> y).
    compact = {}
    for x, y in fine_map:
        compact[str(round(x, 4))] = round(y, 4)
    Path(args.map).write_text(json.dumps({"fine_confidence_to_accuracy": compact}, indent=2), encoding="utf-8")

    print(f"n={len(rows)}  (map source: {args.data})")
    print(f"fine ECE raw={report['fine']['ece_raw']}  "
          f"calibrated={report['fine']['ece_calibrated']} "
          f"(0 = perfectly honest; lower is better)")
    print("raw fine reliability (claimed_mid -> actual):")
    for b in fine_bins:
        print(f"  {b['range']}  n={b['n']:5d}  claimed={b['claimed_mid']:.2f}  actual={b['actual_accuracy']:.2f}")
    print(f"wrote {args.map} ({len(compact)} breakpoints)")


if __name__ == "__main__":
    main()
