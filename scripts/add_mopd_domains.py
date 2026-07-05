#!/usr/bin/env python3
"""Add a 'domains' column to a parquet dataset for MOPD routing validation.

Assigns domain labels to each row so that multiple teachers receive traffic.
Default: alternate between two domains (e.g. "math" and "general") so both
teachers are exercised in every training run.

Usage:
    python scripts/add_mopd_domains.py \\
        --input  /data/gsm8k/train.parquet \\
        --output /data/gsm8k/train_mopd.parquet \\
        --domains math general \\
        --mode alternate

    # Or assign a fixed ratio (70% math, 30% general):
    python scripts/add_mopd_domains.py \\
        --input  /data/gsm8k/train.parquet \\
        --output /data/gsm8k/train_mopd.parquet \\
        --domains math general \\
        --mode ratio --ratio 0.7

    # Verify routing after generation:
    python scripts/add_mopd_domains.py --verify /data/gsm8k/train_mopd.parquet
"""

import argparse
import sys

import pandas as pd


def add_domains_alternate(df: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    """Assign domains by cycling through the list row by row."""
    df = df.copy()
    df["domains"] = [domains[i % len(domains)] for i in range(len(df))]
    return df


def add_domains_ratio(df: pd.DataFrame, domains: list[str], ratio: float) -> pd.DataFrame:
    """Assign first domain to `ratio` fraction of rows, rest to second domain.

    Only supports exactly 2 domains.
    """
    if len(domains) != 2:
        raise ValueError("--mode ratio requires exactly 2 domains")
    df = df.copy()
    cutoff = int(len(df) * ratio)
    df["domains"] = [domains[0]] * cutoff + [domains[1]] * (len(df) - cutoff)
    return df


def verify(path: str) -> None:
    df = pd.read_parquet(path)
    if "domains" not in df.columns:
        print(f"ERROR: no 'domains' column in {path}")
        sys.exit(1)
    counts = df["domains"].value_counts()
    total = len(df)
    print(f"\n  File   : {path}")
    print(f"  Rows   : {total}")
    print(f"  Domain distribution:")
    for domain, count in counts.items():
        print(f"    {domain:<20} {count:>6}  ({100 * count / total:.1f}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", help="Input parquet file")
    parser.add_argument("--output", "-o", help="Output parquet file")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["math", "general"],
        help="Domain labels to assign (default: math general)",
    )
    parser.add_argument(
        "--mode",
        choices=["alternate", "ratio"],
        default="alternate",
        help="alternate: cycle through domains row by row; ratio: split by fraction (default: alternate)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Fraction of rows assigned to the first domain when --mode ratio (default: 0.5)",
    )
    parser.add_argument(
        "--verify",
        metavar="FILE",
        help="Print domain distribution of an existing file and exit",
    )
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --verify is used")

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    if args.mode == "alternate":
        df = add_domains_alternate(df, args.domains)
    else:
        df = add_domains_ratio(df, args.domains, args.ratio)

    df.to_parquet(args.output, index=False)
    print(f"Saved  {len(df)} rows to   {args.output}")
    verify(args.output)


if __name__ == "__main__":
    main()
