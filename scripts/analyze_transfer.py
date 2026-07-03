#!/usr/bin/env python
"""Diagnose cross-corpus transfer from existing prediction artifacts (CPU-only).

Answers the two questions run 2 raised:

1. Why is OneStop low (Spearman ~0.41)? Decompose it: can the model order the
   re-leveled versions of the SAME article (within-article, the construct CLEAR
   never taught), vs order DIFFERENT articles by difficulty (across-article, the
   construct it did learn)? A large across>within gap says construct mismatch,
   not general failure.

2. Is the OOD under-prediction (signed error ~ -0.12..-0.16) just a calibration
   offset? Fit a monotone (isotonic) map on k labels per held-out corpus and
   report in-scale error before/after on the rest. Rank metrics are invariant
   under monotone maps, so this only affects absolute-scale deployment.

    python scripts/analyze_transfer.py --predictions artifacts/student_preds.csv
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from readability.config import load_config
from readability.evaluation import mean_signed_error, rmse, score_predictions
from readability.schema import read_table
from readability.utils import get_logger

log = get_logger("analyze")


def per_corpus_table(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    for (corpus, split), g in df.groupby(["corpus", "split"]):
        rows.append({"corpus": corpus, "split": split,
                     **score_predictions(g[target_col], g["pred"])})
    return pd.DataFrame(rows).sort_values(["split", "corpus"])


def onestop_decomposition(df: pd.DataFrame) -> dict[str, float]:
    """Within-article vs across-article ordering accuracy for OneStopEnglish.

    ids are onestop:{article}:{level}:{chunk}; native_label is the 0/1/2 level.
    Article-level prediction = mean chunk prediction per (article, level).
    """
    os_df = df[df["corpus"] == "onestop"].copy()
    if os_df.empty:
        return {}
    parts = os_df["id"].str.split(":")
    os_df["article"] = parts.str[1]
    lvl = pd.to_numeric(os_df["native_label"], errors="coerce")
    os_df["level"] = lvl
    art = (os_df.groupby(["article", "level"])["pred"].mean().reset_index())

    # within-article: for each article, are its levels ordered correctly?
    n_ok = n_tot = 0
    for _, g in art.groupby("article"):
        if len(g) < 2:
            continue
        for (l1, p1), (l2, p2) in combinations(zip(g["level"], g["pred"]), 2):
            if l1 == l2:
                continue
            n_tot += 1
            n_ok += ((l1 - l2) * (p1 - p2)) > 0
    within = n_ok / n_tot if n_tot else float("nan")

    # across-article: pairs of different articles at (possibly) different levels
    rng = np.random.default_rng(42)
    a = art.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(a)
    i = rng.integers(0, n, size=min(200_000, n * (n - 1)))
    j = rng.integers(0, n, size=len(i))
    keep = (i != j) & (a["article"].to_numpy()[i] != a["article"].to_numpy()[j])
    dl = a["level"].to_numpy()[i][keep] - a["level"].to_numpy()[j][keep]
    dp = a["pred"].to_numpy()[i][keep] - a["pred"].to_numpy()[j][keep]
    valid = dl != 0
    across = float(np.mean((dl[valid] * dp[valid]) > 0)) if valid.sum() else float("nan")

    return {"within_article_acc": float(within), "n_within_pairs": int(n_tot),
            "across_article_acc": across, "n_articles": int(art["article"].nunique())}


def calibration_probe(df: pd.DataFrame, target_col: str, *, k: int = 50,
                      seed: int = 42) -> pd.DataFrame:
    """Per held-out corpus: fit isotonic pred->target on k labels, evaluate
    in-scale error on the remainder. The few-label adaptation cost of deployment."""
    from sklearn.isotonic import IsotonicRegression

    rows = []
    rng = np.random.default_rng(seed)
    for corpus, g in df[df["split"].isin(["ood_corpus", "ood_format"])].groupby("corpus"):
        g = g.dropna(subset=[target_col, "pred"])
        if len(g) < k * 2:
            continue
        idx = rng.permutation(len(g))
        fit, rest = g.iloc[idx[:k]], g.iloc[idx[k:]]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(fit["pred"], fit[target_col])
        calibrated = iso.predict(rest["pred"])
        rows.append({
            "corpus": corpus, "k_labels": k, "n_eval": len(rest),
            "rmse_before": rmse(rest[target_col], rest["pred"]),
            "rmse_after": rmse(rest[target_col], calibrated),
            "signed_before": mean_signed_error(rest[target_col], rest["pred"]),
            "signed_after": mean_signed_error(rest[target_col], calibrated),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--predictions", default="artifacts/student_preds.csv")
    ap.add_argument("--table", default=None)
    ap.add_argument("--target", default="harmonized_difficulty")
    ap.add_argument("--calib-k", type=int, default=50)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    table = read_table(args.table or cfg.data.unified_table)
    preds = pd.read_csv(args.predictions)
    df = table.merge(preds[["id", "pred"]], on="id", how="inner")
    if df.empty:
        raise SystemExit("no id overlap between predictions and the table")

    print("\n=== Per-corpus metrics (held-out rows) ===")
    print(per_corpus_table(df, args.target).to_string(index=False))

    deco = onestop_decomposition(df)
    if deco:
        print("\n=== OneStop decomposition (article-level) ===")
        print(f"  within-article ordering acc : {deco['within_article_acc']:.3f}"
              f"  (n_pairs={deco['n_within_pairs']}) <- construct CLEAR never taught")
        print(f"  across-article ordering acc : {deco['across_article_acc']:.3f}"
              f"  (n_articles={deco['n_articles']})  <- construct the model learned")

    calib = calibration_probe(df, args.target, k=args.calib_k)
    if len(calib):
        print(f"\n=== Few-label calibration probe (isotonic, k={args.calib_k}/corpus) ===")
        print(calib.round(4).to_string(index=False))
        print("  (rank metrics are invariant under monotone maps; this is the "
              "absolute-scale deployment cost)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
