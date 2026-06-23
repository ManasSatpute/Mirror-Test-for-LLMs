"""
Compute the two result tables from labelled Mirror Test runs.

Run mirror_test.py first, label the label_* fields by hand (per the rubric),
then point this at one or more labelled result files:

    python metrics.py results_gpt4o.json results_claude.json results_llama.json

Definitions (kept deliberately simple so they're easy to defend in the viva):
    Hallucination rate   = hallucinated answers / all answers
    Self-detection rate   = errors the model flagged in Layer 2 / all errors
    Self-correction rate  = errors fixed in Layer 3 / errors NOT caught in Layer 2
    Overconfidence Score  = hallucinated AND not self-detected / all answers
                            (your novel metric: wrong and doesn't know it)

A "partial" answer is treated as an error here (it contains a hallucinated
part). If you'd rather exclude partials, change IS_ERROR below — but decide
once, as a team, and write it in the paper.
"""

import json
import sys
from collections import defaultdict


def is_error(label):
    """An answer counts as an error if it's hallucinated or partial."""
    return label in ("hallucinated", "partial")


def summarise(path):
    with open(path) as f:
        data = json.load(f)
    rows = [r for r in data["results"] if "error" not in r]

    per_cat = defaultdict(lambda: {"n": 0, "halluc": 0})
    total = {"n": 0, "halluc": 0, "errors": 0,
             "detected": 0, "corrected": 0, "overconfident": 0}

    for r in rows:
        l1 = r["label_layer1"].strip().lower()
        detected = r["label_self_detected"].strip().lower() == "yes"
        corrected = r["label_self_corrected"].strip().lower() == "yes"
        cat = r["category"]

        total["n"] += 1
        per_cat[cat]["n"] += 1

        if l1 == "hallucinated":
            total["halluc"] += 1
            per_cat[cat]["halluc"] += 1

        if is_error(l1):
            total["errors"] += 1
            if detected:
                total["detected"] += 1
            else:
                # Only answers the model missed in Layer 2 get the nudge chance.
                if corrected:
                    total["corrected"] += 1
                total["overconfident"] += 1  # wrong and didn't catch it

    return data["model"], per_cat, total


def pct(num, den):
    return f"{100 * num / den:5.1f}%" if den else "   n/a"


def main(paths):
    summaries = [summarise(p) for p in paths]
    categories = sorted({c for _, pc, _ in summaries for c in pc})

    # Table 1 — hallucination rate by model and category.
    print("\nTable 1 — Hallucination Rate by Model and Category")
    header = ["Model"] + categories + ["Overall"]
    print("  ".join(f"{h:>14}" for h in header))
    for model, pc, total in summaries:
        cells = [model]
        for c in categories:
            cells.append(pct(pc[c]["halluc"], pc[c]["n"]))
        cells.append(pct(total["halluc"], total["n"]))
        print("  ".join(f"{c:>14}" for c in cells))

    # Table 2 — detection, correction, overconfidence.
    print("\nTable 2 — Self-Detection, Self-Correction, Overconfidence")
    cols = ["Model", "Halluc", "Detect", "Correct", "Overconf"]
    print("  ".join(f"{c:>14}" for c in cols))
    for model, _, t in summaries:
        row = [
            model,
            pct(t["halluc"], t["n"]),
            pct(t["detected"], t["errors"]),
            pct(t["corrected"], t["errors"] - t["detected"]),
            pct(t["overconfident"], t["n"]),
        ]
        print("  ".join(f"{c:>14}" for c in row))
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metrics.py results_*.json")
        sys.exit(1)
    main(sys.argv[1:])
