"""
Mirror Test Evaluation Metrics
"""

import json
import sys

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from tabulate import tabulate
from typing import NamedTuple


# Stats
class CategoryStats(NamedTuple):
    """Hallucination counts for a single (model, category) pair."""
    total: int
    hallucinated: int


class ModelStats(NamedTuple):
    """Aggregate counts for a single model across all categories."""
    total: int
    hallucinated: int
    errors: int
    detected: int
    corrected: int
    overconfident: int


class ModelSummary(NamedTuple):
    """Full summary for one result file: model name, per-category and aggregate stats."""
    model_name: str
    per_category: dict[str, CategoryStats]
    totals: ModelStats


# Helper Methods
def is_error(label: str) -> bool:
    """Return True if the label represents a hallucination error (hallucinated or partial)."""
    return label in {"hallucinated", "partial"}


def pct(numerator: int, denominator: int) -> str:
    """Format a ratio as a fixed-width percentage string, or '   n/a' if denominator is zero."""
    if not denominator:
        return "   n/a"
    return f"{100 * numerator / denominator:5.1f}%"


# Core logic
def load_results(path: str | Path) -> tuple[str, list[dict]]:
    """Load a labelled JSON result file; return the model name and error-free rows."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        data = json.load(fh)
    clean_rows = [row for row in data["results"] if "error" not in row]
    return data["model"], clean_rows


def compute_stats(rows: list[dict]) -> tuple[dict[str, CategoryStats], ModelStats]:
    """Compute per-category and aggregate metrics from cleaned, labelled result rows."""
    raw_cats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "halluc": 0})
    t = {"n": 0, "halluc": 0, "errors": 0, "detected": 0, "corrected": 0, "overconfident": 0}

    for row in rows:
        label     = row["label_layer1"].strip().lower()
        detected  = row["label_self_detected"].strip().lower() == "yes"
        corrected = row["label_self_corrected"].strip().lower() == "yes"
        cat       = row["category"]

        t["n"] += 1
        raw_cats[cat]["n"] += 1

        if label == "hallucinated":
            t["halluc"] += 1
            raw_cats[cat]["halluc"] += 1

        if is_error(label):
            t["errors"] += 1
            if detected:
                t["detected"] += 1
            else:
                # Eligible for Layer 3 correction only if missed in Layer 2.
                if corrected:
                    t["corrected"] += 1
                t["overconfident"] += 1  # wrong and didn't catch it

    per_category = {
        cat: CategoryStats(total=v["n"], hallucinated=v["halluc"])
        for cat, v in raw_cats.items()
    }
    totals = ModelStats(
        total=t["n"], hallucinated=t["halluc"], errors=t["errors"],
        detected=t["detected"], corrected=t["corrected"], overconfident=t["overconfident"],
    )
    return per_category, totals


def summarise(path: str | Path) -> ModelSummary:
    """Load and summarise one result file into a ModelSummary."""
    model_name, rows = load_results(path)
    per_category, totals = compute_stats(rows)
    return ModelSummary(model_name=model_name, per_category=per_category, totals=totals)


# Table builders
def build_table1(summaries: list[ModelSummary], categories: list[str]) -> None:
    """Print Table 1: hallucination rate per model per category, plus an overall column."""
    rows = []
    for s in summaries:
        row = [s.model_name]
        for cat in categories:
            stats = s.per_category.get(cat, CategoryStats(0, 0))
            row.append(pct(stats.hallucinated, stats.total))
        row.append(pct(s.totals.hallucinated, s.totals.total))
        rows.append(row)

    print("\nTable 1 — Hallucination Rate by Model and Category\n")
    print(tabulate(rows, headers=["Model"] + categories + ["Overall"],
                   tablefmt="rounded_grid", stralign="center", numalign="center"))


def build_table2(summaries: list[ModelSummary]) -> None:
    """Print Table 2: hallucination, self-detection, self-correction, and overconfidence rates."""
    rows = []
    for s in summaries:
        t = s.totals
        rows.append([
            s.model_name,
            pct(t.hallucinated, t.total),
            pct(t.detected, t.errors),
            pct(t.corrected, t.errors - t.detected),
            pct(t.overconfident, t.total),
        ])

    print("\nTable 2 — Self-Detection, Self-Correction, Overconfidence\n")
    print(tabulate(rows,
                   headers=["Model", "Hallucination", "Self-Detection",
                             "Self-Correction", "Overconfidence"],
                   tablefmt="rounded_grid", stralign="center", numalign="center"))


# Main Method
def main(paths: list[str]) -> None:
    """Summarise all result files and print both metric tables."""
    summaries = [summarise(p) for p in paths]
    all_categories = sorted({cat for s in summaries for cat in s.per_category})
    build_table1(summaries, all_categories)
    build_table2(summaries)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metrics.py results_*.json")
        sys.exit(1)
    main(sys.argv[1:])