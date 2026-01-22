#!/usr/bin/env python3
"""
Aggregate metrics.json files under results/* into a Markdown summary.

Usage:
    python scripts/aggregate_results.py \
        --results_dir results \
        --out_md results/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate metrics.json into Markdown.")
    p.add_argument("--results_dir", type=str, default="results", help="Root directory containing run folders.")
    p.add_argument("--out_md", type=str, default="", help="Optional path to save markdown table.")
    return p.parse_args()


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    if x is None:
        return ""
    return str(x)


def collect_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        data = load_metrics(metrics_path)
        run_name = data.get("run_name") or metrics_path.parent.name
        best = data.get("best") or {}
        last = data.get("last") or {}
        primary = data.get("best_primary")
        stats = best if best else last
        rows.append(
            {
                "run": run_name,
                "epoch": stats.get("epoch"),
                "best_primary": primary,
                "val_spearman": stats.get("val_spearman"),
                "val_pearson": stats.get("val_pearson"),
                "val_loss": stats.get("val_loss"),
                "train_loss": stats.get("train_loss"),
                "path": str(metrics_path.parent),
            }
        )
    return rows


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = ["run", "epoch", "best_primary", "val_spearman", "val_pearson", "val_loss", "train_loss", "path"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.results_dir)
    rows = collect_rows(root)
    md = to_markdown(rows)
    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"[OK] Wrote summary to {out_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
