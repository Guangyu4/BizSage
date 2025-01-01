"""
Aggregate per-query judge scores into summary CSV tables.

Usage:
    python aggregate_scores.py --eval-dir eval_out/ --output scores.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--output", default="scores.csv")
    args = p.parse_args()

    rows = []
    for task_type in ["survey", "idea_formulation", "research_plan"]:
        task_dir = os.path.join(args.eval_dir, task_type)
        if not os.path.isdir(task_dir):
            continue
        dim_totals = defaultdict(list)
        for fname in sorted(os.listdir(task_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(task_dir, fname), "r") as f:
                data = json.load(f)
            scores = data.get("scores", {})
            for dim, val in scores.items():
                dim_totals[dim].append(val)
            rows.append({
                "task": task_type,
                "paper_id": data.get("paper_id", fname),
                **scores,
                "total": data.get("total", sum(scores.values())),
            })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
