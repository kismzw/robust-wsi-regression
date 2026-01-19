#!/usr/bin/env python3
"""
make_splits.py

Create a reproducible, group-aware train/val split for image->high-dimensional regression.

Input metadata CSV must contain at least:
  - path
  - group_id
  - y0 ... y(K-1)  (not strictly required for splitting, but recommended)

Output CSV will contain:
  - path
  - group_id
  - split  (train|val)

Example:
  python make_splits.py \
    --metadata_csv data/metadata.csv \
    --out_csv splits/split_seed42.csv \
    --seed 42 \
    --val_ratio 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create group-aware train/val split CSV.")
    p.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV.")
    p.add_argument("--out_csv", type=str, required=True, help="Output split CSV path.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio by groups.")
    p.add_argument(
        "--group_col",
        type=str,
        default="group_id",
        help="Column name for group identifier (default: group_id).",
    )
    p.add_argument(
        "--path_col",
        type=str,
        default="path",
        help="Column name for image path (default: path).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    meta_path = Path(args.metadata_csv)
    out_path = Path(args.out_csv)

    if not meta_path.exists():
        print(f"[ERROR] metadata_csv not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    if not (0.0 < args.val_ratio < 1.0):
        print("[ERROR] val_ratio must be between 0 and 1 (exclusive).", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(meta_path)

    for col in (args.path_col, args.group_col):
        if col not in df.columns:
            print(
                f"[ERROR] Required column '{col}' not found in metadata CSV. "
                f"Available columns: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Basic sanity checks
    if df[args.path_col].isna().any():
        n = int(df[args.path_col].isna().sum())
        print(f"[ERROR] Found {n} rows with missing '{args.path_col}'.", file=sys.stderr)
        sys.exit(1)

    if df[args.group_col].isna().any():
        n = int(df[args.group_col].isna().sum())
        print(f"[ERROR] Found {n} rows with missing '{args.group_col}'.", file=sys.stderr)
        sys.exit(1)

    # Ensure stable ordering before sampling groups
    df = df.reset_index(drop=True)

    # Sample groups into val set reproducibly
    groups = df[args.group_col].astype(str)
    unique_groups = groups.drop_duplicates().tolist()

    n_groups = len(unique_groups)
    n_val_groups = max(1, int(round(n_groups * args.val_ratio)))

    # Reproducible shuffle of groups
    unique_groups_series = pd.Series(unique_groups)
    unique_groups_shuffled = unique_groups_series.sample(
        frac=1.0, random_state=args.seed
    ).tolist()

    val_groups = set(unique_groups_shuffled[:n_val_groups])

    split = ["val" if g in val_groups else "train" for g in groups]
    out_df = pd.DataFrame(
        {
            args.path_col: df[args.path_col].astype(str),
            args.group_col: groups,
            "split": split,
        }
    )

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_path, index=False)

    # Print summary
    n_train = int((out_df["split"] == "train").sum())
    n_val = int((out_df["split"] == "val").sum())
    n_train_groups = int(out_df.loc[out_df["split"] == "train", args.group_col].nunique())
    n_val_groups = int(out_df.loc[out_df["split"] == "val", args.group_col].nunique())

    print("[OK] Wrote split file:", out_path)
    print(f"      Total samples: {len(out_df)}")
    print(f"      Train samples: {n_train}  | groups: {n_train_groups}")
    print(f"      Val samples:   {n_val}  | groups: {n_val_groups}")
    print(f"      Seed: {args.seed} | val_ratio (groups): {args.val_ratio}")

    # Guardrail: check for leakage
    train_groups = set(out_df.loc[out_df["split"] == "train", args.group_col].unique())
    leaked = train_groups.intersection(val_groups)
    if leaked:
        print(f"[WARN] Group leakage detected (should be empty): {list(leaked)[:5]}", file=sys.stderr)


if __name__ == "__main__":
    main()
