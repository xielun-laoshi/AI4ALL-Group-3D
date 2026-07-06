#!/usr/bin/env python
"""One-command LOCO multi-gold experiment (the label-diversity headline).

For each fold (one corpus held out) x seed x variant: rebuild the fold's table
under artifacts/loco_{fold}/, train the student on the remaining gold corpora,
score the held-out corpus, and additionally score the zero-training hybrid
rank-blend of each variant's mean predictions. Reports Spearman mean +/- std per
fold x variant and a paired bootstrap of each variant against the gold baseline
on the same held-out items.

    python scripts/run_loco.py --seeds 42 43 44
    python scripts/run_loco.py --folds onestop --variants gold gold_ff
    # debug/smoke: subsample + tiny backbone via trailing overrides
    python scripts/run_loco.py --folds onestop --seeds 42 --max-rows-per-corpus 300 \
        model.backbone=prajjwal1/bert-tiny train.epochs=1

Per-fold tables + per-run predictions are written under artifacts/loco_{fold}/
(nothing clobbers), so analyze_transfer can be pointed at MATCHING table+preds:
    python scripts/analyze_transfer.py --table artifacts/loco_onestop/corpus.csv \
        --predictions artifacts/loco_onestop/preds_gold_s42.csv --hybrid --formula-baseline
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import data_preprocessing
from readability.ablation import paired_bootstrap_diff, run_variant
from readability.config import load_config
from readability.evaluation import spearman
from readability.external import difficulty_proxy
from readability.schema import read_table
from readability.utils import artifacts_dir, get_logger
from analyze_transfer import rank_blend

log = get_logger("loco")

# variant -> config overrides (train table is always the fold's gold table).
VARIANTS: dict[str, list[str]] = {
    "gold":    [],                                  # multi-gold baseline
    "gold_ff": ["model.use_formula_feature=true"],  # + formula proxy in the head
    # + pairwise head over pair-corpus articles (wiki-pairs construct test:
    # include wikipair via --corpora; with the head OFF those rows are inert,
    # so gold vs gold_pw on the same table isolates the pair supervision).
    "gold_pw": ["model.use_pairwise_head=true"],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--folds", nargs="+", default=["onestop", "cefr"],
                    help="each fold holds out this corpus and trains on the rest")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--variants", nargs="+", default=["gold", "gold_ff"], choices=list(VARIANTS))
    ap.add_argument("--corpora", nargs="+", default=None,
                    help="override the corpora in each fold's table, e.g. "
                         "clear onestop cefr wikipair (wikipair adds pair supervision)")
    ap.add_argument("--hybrid-weight", type=float, default=0.5,
                    help="model weight in the zero-training rank-blend scored per variant")
    ap.add_argument("--max-rows-per-corpus", type=int, default=0,
                    help="debug: subsample each corpus (0 = all rows)")
    ap.add_argument("overrides", nargs="*",
                    help="extra dotted overrides applied to every run, e.g. train.epochs=2")
    args = ap.parse_args(argv)

    rows: list[dict] = []
    for fold in args.folds:
        fold_dir = artifacts_dir() / f"loco_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        table_path = fold_dir / "corpus.csv"
        # per-fold table at its own path: no stale-table aliasing between folds
        dp_args = ["--config", args.config, "--skip-download",
                   "--holdout-corpora", fold, "--out", str(table_path)]
        if args.corpora:
            dp_args += ["--corpora", *args.corpora]
        data_preprocessing.main(dp_args)
        table = read_table(table_path)
        if args.max_rows_per_corpus:
            table = (table.groupby("corpus", group_keys=False)
                     .apply(lambda g: g.sample(n=min(args.max_rows_per_corpus, len(g)),
                                               random_state=0))
                     .reset_index(drop=True))

        preds_by: dict[str, np.ndarray] = {}
        target = None
        for variant in args.variants:
            per_seed = []
            for seed in args.seeds:
                cfg = load_config(args.config,
                                  overrides=VARIANTS[variant] + list(args.overrides)
                                  + [f"train.seed={seed}"])
                metrics, ids, pr, tgt = run_variant(cfg, table, table,
                                                    holdout_split="ood_corpus")
                pd.DataFrame({"id": ids, "pred": pr}).to_csv(
                    fold_dir / f"preds_{variant}_s{seed}.csv", index=False)
                rows.append({"fold": fold, "variant": variant, "seed": seed, **metrics})
                log.info("[%s %s seed=%d] spearman=%.4f", fold, variant, seed, metrics["spearman"])
                per_seed.append(pr)
                target = tgt
            preds_by[variant] = np.nanmean(np.vstack(per_seed), axis=0)

        # zero-training hybrid of each variant's mean preds with the formula proxy
        hold = table[table["split"] == "ood_corpus"]
        proxy = hold["text"].astype(str).map(difficulty_proxy).to_numpy()
        for base in list(preds_by):
            name = f"{base}+hybrid"
            preds_by[name] = rank_blend(preds_by[base], proxy, w=args.hybrid_weight)
            rows.append({"fold": fold, "variant": name, "seed": -1,
                         "spearman": spearman(target, preds_by[name]),
                         "rmse": float("nan"), "n": int(np.isfinite(target).sum())})

        if "gold" in preds_by:
            print(f"\n=== fold {fold}: vs gold baseline (paired bootstrap, Spearman; "
                  f"+ => variant better, * p<0.05) ===")
            for name, pr in preds_by.items():
                if name == "gold":
                    continue
                s = paired_bootstrap_diff(target, pr, preds_by["gold"], higher_is_better=True)
                star = "*" if s["p_full_not_better"] < 0.05 else " "
                print(f"  {name:<16s} delta={s['delta']:+.4f}  "
                      f"95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  "
                      f"p={s['p_full_not_better']:.3f} {star}")

    df = pd.DataFrame(rows)
    out = artifacts_dir() / "loco_results.csv"
    df.to_csv(out, index=False)
    print("\n=== LOCO summary (Spearman on the held-out corpus; mean +/- std over seeds) ===")
    summ = (df.groupby(["fold", "variant"])
            .agg(spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
                 runs=("seed", "count")).reset_index())
    print(summ.to_string(index=False))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
