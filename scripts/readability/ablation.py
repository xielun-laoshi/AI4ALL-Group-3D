"""Ablations (framework Phase 8): prove each design choice *caused* the gain.

Every variant toggles ONE thing (a config override and/or which training table is
used), everything else fixed, and is scored on the cross-corpus holdout. The delta
vs the full model -- with a paired bootstrap CI and a one-sided p -- is what turns
"we improved on the winner" from an assertion into an attributable result.

run_variant + the stats here are torch-light (Trainer imported lazily), so the
significance/aggregation logic tests without a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import rmse, spearman
from .utils import get_logger

log = get_logger("ablation")

# variant -> {overrides: dotted config flags, train_table: "train_pool" | "gold"}.
# Each turns exactly one component off relative to "full".
ABLATIONS: dict[str, dict] = {
    "full":             {"overrides": [], "train_table": "train_pool"},
    "no_pairwise":      {"overrides": ["model.use_pairwise_head=false"], "train_table": "train_pool"},
    "no_source_offset": {"overrides": ["model.use_source_offset=false"], "train_table": "train_pool"},
    "no_confidence_wt": {"overrides": ["train.confidence_weighting=false"], "train_table": "train_pool"},
    "no_pseudo":        {"overrides": [], "train_table": "gold"},   # gold only (data lever)
    # additive variant: two-stage pretrain->finetune. A NEGATIVE delta vs full
    # (single-pass) here means two-stage helped -> consider making it the default.
    "two_stage":        {"overrides": ["train.two_stage=true"], "train_table": "train_pool"},
    # additive variant: formula/model hybrid -- standardized proxy concat into the
    # head (per-corpus floors: proxy 0.79 on CEFR, 0.97 within-article OneStop).
    # A NEGATIVE delta vs full means the feature helped -> consider defaulting on.
    "formula_feature":  {"overrides": ["model.use_formula_feature=true"], "train_table": "train_pool"},
}


def run_variant(cfg, train_df: pd.DataFrame, eval_df: pd.DataFrame, *,
                holdout_split: str = "ood_corpus",
                target_col: str = "harmonized_difficulty"):
    """Train one variant and score it on the cross-corpus holdout.

    Returns (metrics dict, ids, preds, targets) -- preds/targets are kept so the
    runner can run a *paired* bootstrap between variants on the same items.
    """
    from .training import fit_student

    trainer = fit_student(cfg, train_df, target_col=target_col)

    hold = eval_df[eval_df["split"] == holdout_split].copy()
    pr = trainer.predict(hold).set_index("id").reindex(hold["id"].astype(str))["pred"].to_numpy()
    tgt = pd.to_numeric(hold[target_col], errors="coerce").to_numpy()
    metrics = {"spearman": spearman(tgt, pr), "rmse": rmse(tgt, pr),
               "n": int(np.isfinite(tgt).sum())}
    return metrics, hold["id"].astype(str).tolist(), pr, tgt


def paired_bootstrap_diff(target, pred_full, pred_variant, metric=spearman, *,
                          higher_is_better: bool = True, n_boot: int = 2000,
                          seed: int = 42) -> dict[str, float]:
    """Paired bootstrap of (full - variant) on the SAME holdout items. Positive
    delta => the full model is better. ``p_full_not_better`` is the one-sided
    bootstrap p (fraction of resamples where the delta <= 0)."""
    t = np.asarray(target, float); a = np.asarray(pred_full, float); b = np.asarray(pred_variant, float)
    m = np.isfinite(t) & np.isfinite(a) & np.isfinite(b)
    t, a, b = t[m], a[m], b[m]
    n = len(t)
    rng = np.random.default_rng(seed)

    def delta(idx):
        ma, mb = metric(t[idx], a[idx]), metric(t[idx], b[idx])
        return (ma - mb) if higher_is_better else (mb - ma)

    point = delta(np.arange(n))
    boots = np.array([delta(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_full_not_better": float(np.mean(boots <= 0))}


def aggregate(rows: list[dict]) -> pd.DataFrame:
    """Mean +/- std per variant across seeds, for the reported metrics."""
    df = pd.DataFrame(rows)
    agg = df.groupby("variant").agg(
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        seeds=("seed", "nunique")).reset_index()
    return agg.sort_values("spearman_mean", ascending=False).reset_index(drop=True)
