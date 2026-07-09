#!/usr/bin/env python
"""Mechanistic confirmation for the wiki-pairs experiment (CPU-only, no training).

The LOCO headline says gold_pw lifts held-out OneStop Spearman +0.139. This tool
answers WHERE that lift comes from: it averages each variant's per-seed LOCO
predictions and prints the OneStop within-/across-article decomposition
side-by-side, so `gold` (wikipair inert) vs `gold_pw` (pair supervision on) shows
whether the gain lands on the WITHIN-article re-leveling construct wiki-pairs
targets (toward the formula's ~0.97) without an across-article regression -- the
experiment's actual success criterion, which the aggregate Spearman can't show.

    python scripts/decompose_pairs.py --fold-dir artifacts/loco_onestop \
        --variants gold gold_pw

Predictions are the per-seed files run_loco.py writes: preds_{variant}_s{seed}.csv
(columns id,pred). The table defaults to the fold's own corpus.csv (the OneStop
rows are identical across variants, so any fold table with them works).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from readability.evaluation import spearman
from readability.schema import read_table
from analyze_transfer import onestop_decomposition


def averaged_preds(fold_dir: str | Path, variant: str) -> tuple[pd.DataFrame, int]:
    """Mean prediction per id over a variant's per-seed preds_{variant}_s*.csv."""
    files = sorted(glob.glob(str(Path(fold_dir) / f"preds_{variant}_s*.csv")))
    if not files:
        raise SystemExit(f"no preds_{variant}_s*.csv in {fold_dir}")
    wide = pd.concat([pd.read_csv(f).set_index("id")["pred"] for f in files], axis=1)
    mean = wide.mean(axis=1).rename("pred").reset_index()
    return mean, len(files)


def decompose_variant(table: pd.DataFrame, preds: pd.DataFrame,
                      target_col: str) -> dict[str, float]:
    """OneStop aggregate Spearman (the LOCO headline metric) + the within/across
    decomposition, for one variant's averaged predictions."""
    df = table.merge(preds, on="id", how="inner")
    if df.empty:
        raise SystemExit("no id overlap between predictions and the table")
    os_df = df[df["corpus"] == "onestop"]
    out = {"spearman": spearman(os_df[target_col], os_df["pred"]), "n": len(os_df)}
    out.update(onestop_decomposition(df))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold-dir", default="artifacts/loco_onestop")
    ap.add_argument("--table", default=None, help="default: <fold-dir>/corpus.csv")
    ap.add_argument("--variants", nargs="+", default=["gold", "gold_pw"])
    ap.add_argument("--target", default="harmonized_difficulty")
    ap.add_argument("--no-formula", action="store_true",
                    help="skip the free formula-proxy goalpost row (~0.97 within-article)")
    args = ap.parse_args(argv)

    table = read_table(args.table or str(Path(args.fold_dir) / "corpus.csv"))

    res: dict[str, dict[str, float]] = {}
    for v in args.variants:
        preds, n_seed = averaged_preds(args.fold_dir, v)
        res[v] = {"seeds": n_seed, **decompose_variant(table, preds, args.target)}

    order = list(args.variants)
    if not args.no_formula:
        # the goalpost: the free proxy owns within-article re-leveling (~0.97), the
        # construct wiki-pairs targets. Computed from text, so no preds file needed.
        from readability.external import difficulty_proxy
        os_rows = table[table["corpus"] == "onestop"].copy()
        os_rows["pred"] = os_rows["text"].astype(str).map(difficulty_proxy)
        res["formula"] = {"seeds": 0,
                          **decompose_variant(table, os_rows[["id", "pred"]], args.target)}
        order.append("formula")

    print("\n=== OneStop decomposition by variant "
          "(mean over seeds; within-article = the wiki-pairs construct) ===")
    hdr = f"{'variant':<10}{'seeds':>6}{'spearman':>10}{'within':>9}{'across':>9}{'n_within':>10}"
    print(hdr)
    print("-" * len(hdr))
    for v in order:
        r = res[v]
        print(f"{v:<10}{r['seeds']:>6}{r['spearman']:>10.4f}"
              f"{r.get('within_article_acc', float('nan')):>9.3f}"
              f"{r.get('across_article_acc', float('nan')):>9.3f}"
              f"{r.get('n_within_pairs', 0):>10}")

    # delta of the last variant vs the first (typically gold_pw vs gold)
    if len(args.variants) >= 2:
        a, b = args.variants[0], args.variants[-1]
        ra, rb = res[a], res[b]
        print(f"\ndelta ({b} - {a}):  spearman {rb['spearman'] - ra['spearman']:+.4f}"
              f"   within {rb.get('within_article_acc', np.nan) - ra.get('within_article_acc', np.nan):+.3f}"
              f"   across {rb.get('across_article_acc', np.nan) - ra.get('across_article_acc', np.nan):+.3f}")
        print("  success = within moves up toward the formula's ~0.97 with across "
              "roughly held (no re-leveling gained at the cost of general ordering)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
