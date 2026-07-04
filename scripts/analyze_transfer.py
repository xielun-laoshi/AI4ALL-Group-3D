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
    """Per corpus: fit isotonic pred->target on k labels, evaluate in-scale error
    on the remainder. The few-label adaptation cost of deployment.

    Groups by CORPUS over all predicted rows, NOT by split: the on-disk table is
    rebuilt per LOCO fold, so analyzing fold-A predictions against a fold-B table
    leaves the held-out rows' split labels stale ('train'), and a split filter
    silently empties the probe (the bug hit on 2026-07-03). Staleness is warned
    about instead; in-gold corpora just show a ~0 offset."""
    from sklearn.isotonic import IsotonicRegression

    rows = []
    rng = np.random.default_rng(seed)
    stale = df[df["split"] == "train"]
    if len(stale):
        log.warning("predictions exist for %d train-split rows (%s) -- the on-disk "
                    "table likely doesn't match the run that produced these "
                    "predictions; split labels may be stale",
                    len(stale), sorted(stale["corpus"].unique()))
    for corpus, g in df.groupby("corpus"):
        g = g.dropna(subset=[target_col, "pred"])
        if len(g) < k * 2:
            log.info("calibration: skipping %s (%d rows < %d)", corpus, len(g), 2 * k)
            continue
        idx = rng.permutation(len(g))
        fit, rest = g.iloc[idx[:k]], g.iloc[idx[k:]]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(fit["pred"], fit[target_col])
        calibrated = iso.predict(rest["pred"])
        rows.append({
            "corpus": corpus, "splits": "+".join(sorted(g["split"].astype(str).unique())),
            "k_labels": k, "n_eval": len(rest),
            "rmse_before": rmse(rest[target_col], rest["pred"]),
            "rmse_after": rmse(rest[target_col], calibrated),
            "signed_before": mean_signed_error(rest[target_col], rest["pred"]),
            "signed_after": mean_signed_error(rest[target_col], calibrated),
        })
    return pd.DataFrame(rows)


SCALE_FREE_COLS = ["corpus", "split", "spearman", "kendall", "pairwise_acc", "rank_rmse", "n"]


def rank_blend(model_scores, formula_scores, w: float = 0.5) -> np.ndarray:
    """Zero-training hybrid: blend the percentile ranks of the model's scores and
    the formula proxy (w = model weight). Ranks are computed over the scored pool,
    which mirrors deployment (grading a collection). Motivated by complementary
    strengths: model wins on CLEAR + across-article; proxy wins on CEFR +
    within-article (0.97)."""
    rm = pd.Series(np.asarray(model_scores, float)).rank(pct=True)
    rf = pd.Series(np.asarray(formula_scores, float)).rank(pct=True)
    return (w * rm + (1.0 - w) * rf).to_numpy()


def formula_baseline(table: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Score the free formula proxy (difficulty_proxy: sentence length + word
    length) as if it were a model, per corpus. This is the OOD formula floor,
    measured on each held-out corpus ITSELF -- the number a learned model must
    beat there (the CLEAR-measured floor doesn't transfer automatically)."""
    from readability.external import difficulty_proxy

    fb = table.dropna(subset=[target_col]).copy()
    fb["pred"] = fb["text"].astype(str).map(difficulty_proxy)
    return fb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--predictions", default="artifacts/student_preds.csv")
    ap.add_argument("--table", default=None)
    ap.add_argument("--target", default="harmonized_difficulty")
    ap.add_argument("--calib-k", type=int, default=50)
    ap.add_argument("--formula-baseline", action="store_true",
                    help="also score the free formula proxy per corpus (no model needed)")
    ap.add_argument("--hybrid", action="store_true",
                    help="score the zero-training rank-ensemble of model + formula proxy")
    ap.add_argument("--hybrid-weight", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                    help="model weight(s) w in the blend (1-w goes to the proxy); "
                         "multiple values sweep in one run")
    args = ap.parse_args(argv)

    from pathlib import Path

    cfg = load_config(args.config)
    table = read_table(args.table or cfg.data.unified_table)

    if not Path(args.predictions).exists():
        if not args.formula_baseline:
            raise SystemExit(f"{args.predictions} not found "
                             f"(pass --formula-baseline to run without model predictions)")
        df = pd.DataFrame()
    else:
        preds = pd.read_csv(args.predictions)
        df = table.merge(preds[["id", "pred"]], on="id", how="inner")
        if df.empty:
            raise SystemExit("no id overlap between predictions and the table")

    if args.formula_baseline:
        fb = formula_baseline(table, args.target)
        print("\n=== FORMULA floor per corpus (free proxy; rank metrics only) ===")
        print(per_corpus_table(fb, args.target)[SCALE_FREE_COLS].to_string(index=False))
        fdeco = onestop_decomposition(fb)
        if fdeco:
            print(f"  formula on OneStop: within-article acc {fdeco['within_article_acc']:.3f}"
                  f" | across-article acc {fdeco['across_article_acc']:.3f}")

    if df.empty:
        return 0

    print("\n=== Per-corpus metrics (held-out rows) ===")
    print(per_corpus_table(df, args.target).to_string(index=False))

    deco = onestop_decomposition(df)
    if deco:
        print("\n=== OneStop decomposition (article-level) ===")
        print(f"  within-article ordering acc : {deco['within_article_acc']:.3f}"
              f"  (n_pairs={deco['n_within_pairs']}) <- construct CLEAR never taught")
        print(f"  across-article ordering acc : {deco['across_article_acc']:.3f}"
              f"  (n_articles={deco['n_articles']})  <- construct the model learned")

    if args.hybrid:
        from readability.external import difficulty_proxy

        proxy = df["text"].astype(str).map(difficulty_proxy)
        model_pred = df["pred"].copy()
        for w in args.hybrid_weight:
            hy = df.copy()
            hy["pred"] = rank_blend(model_pred, proxy, w=w)
            print(f"\n=== HYBRID rank-ensemble (w_model={w:.2f}; ranks over the scored pool) ===")
            print(per_corpus_table(hy, args.target)[SCALE_FREE_COLS].to_string(index=False))
            hdeco = onestop_decomposition(hy)
            if hdeco:
                print(f"  hybrid on OneStop: within-article acc {hdeco['within_article_acc']:.3f}"
                      f" | across-article acc {hdeco['across_article_acc']:.3f}")

    calib = calibration_probe(df, args.target, k=args.calib_k)
    if len(calib):
        print(f"\n=== Few-label calibration probe (isotonic, k={args.calib_k}/corpus) ===")
        print(calib.round(4).to_string(index=False))
        print("  (rank metrics are invariant under monotone maps; this is the "
              "absolute-scale deployment cost)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
