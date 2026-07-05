# LLM Text Readability Grading

Predict how hard a text is to read, in a way that **generalizes beyond the corpus
it was trained on**. Built on the CLEAR corpus (CommonLit Readability Prize),
refactored from the CLRP 1st-place teacher/student pipeline with three choices
inverted (diverse data, cross-corpus evaluation, human-anchored labels) and three
kept (SE-filtered pseudo-labeling, pretrain→finetune, teacher/student).

> **Status — measured (July 2026).** Full pipeline implemented and run on GPU:
> teacher/student pseudo-labeling, multi-seed LOCO cross-corpus experiments,
> paired-bootstrap ablations, formula/model hybrid. Key numbers below; full log
> in [docs/results-2026-07-02.md](docs/results-2026-07-02.md).

## Headline results (3 seeds, paired bootstrap; Spearman)

| Evaluation | Formula baseline | This system | Note |
|---|---|---|---|
| In-corpus (CLEAR val) | 0.49 | **0.89** | roberta-base regressor |
| Cross-corpus → CEFR | 0.79 | **0.845** | multi-gold + zero-training rank-blend |
| Cross-corpus → OneStop | 0.40 | **0.48 ± 0.02** | multi-gold (blend n.s. here) |

- **Label diversity is the lever:** adding ONE human-labeled corpus lifts the
  held-out corpus +0.106 (OneStop) / +0.040 (CEFR) — ~4–11× any architecture knob.
- **Input diversity alone is inert:** pseudo-labeling a diverse 55k pool with a
  single-corpus teacher moved cross-corpus Spearman ~0 (CI straddles zero) —
  the teacher only re-injects its own label function.
- **Integration point matters:** the same formula signal *hurts* as a training
  feature (−0.024*) but *wins* as an inference-time rank ensemble (+0.053*).
- Residual OneStop gap is a **construct gap** (ordering re-leveled versions of
  the *same* article — present in neither training corpus): model orders
  *different* articles at 0.785 while the formula hits 0.97 *within* articles.

## Layout

```
data/                  raw downloads ONLY (git-ignored; fetched at runtime)
artifacts/             derived outputs: tables, predictions, checkpoints (git-ignored)
configs/               experiment configs; every knob is a value, not a code edit
scripts/               all Python — reproduction entry points + support lib
  data_preprocessing.py  download -> unified schema -> harmonize -> splits
  build_external_pool.py stream free text -> window -> diversity-select (Phase 4a)
  train_teacher.py       teacher ensemble on gold (Phase 4b)
  pseudo_label.py        SE/disagreement/dedup-filtered pseudo-labels (Phase 4c)
  train.py               student training (single-pass or two-stage; --tag names runs)
  evaluate.py            floor/ceiling/baseline bracket + per-split scoring
  analyze_transfer.py    per-corpus diagnostics, formula floor, hybrid blend, calibration
  run_ablations.py       component ablations, paired-bootstrap significance
  run_loco.py            ONE-COMMAND leave-one-corpus-out experiment (folds x seeds)
  readability/           support library (schema, config, data, model, training,
                         evaluation, pseudolabel, ablation, external, utils)
tests/                 pytest suite (28 tests)
```

The repo is a **recipe, not a data dump**: `data/`, `artifacts/`, and `runs/` are
git-ignored and regenerated from code + config. All corpora are free (CLEAR xlsx
auto-downloaded; OneStopEnglish + CEFR levelled texts fetched by the prep script).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU build

python scripts/data_preprocessing.py     # all 3 corpora; onestop+cefr held out
python scripts/train.py                  # student on gold -> artifacts/student_preds.csv
python scripts/evaluate.py --predictions artifacts/student_preds.csv --group-col corpus

# the headline experiment: LOCO multi-gold x seeds x variants + hybrid + significance
python scripts/run_loco.py --seeds 42 43 44

# diagnostics on any predictions (formula floor, hybrid sweep, calibration probe)
python scripts/analyze_transfer.py --table artifacts/loco_onestop/corpus.csv \
    --predictions artifacts/loco_onestop/preds_gold_s42.csv --hybrid --formula-baseline
```

`pytest` runs the suite. Notes: default backbone is `roberta-base`
(deberta-v3 NaNs under transformers ≥ ~4.47); config defaults are
**evidence-annotated** from the ablations (pairwise head off, offset on for
multi-gold — see `configs/default.yaml`).

## The evaluation harness (why a model number means anything)

Every result is bracketed between references so a score is never a bare number:

- **Floor of usefulness** — classic formulas. Measured *per corpus* by
  `analyze_transfer --formula-baseline` (CLEAR 0.49, OneStop 0.40, CEFR 0.79 —
  the CEFR floor is why the hybrid exists). On CLEAR, formulas explain only
  ~27–33% of human-label variance.
- **Floor of achievability** — mean per-item `BT s.e.` (~0.49), a *soft* label-noise
  reference.
- **Human-level comparator** — CLRP winners at ~0.45 RMSE / ~0.95 Spearman (their
  prediction columns are absent from the public CLEAR file; documented values).

Rank metrics (Spearman, Kendall, pairwise accuracy, rank-RMSE) are the headline
because they transfer across corpora; raw RMSE/MAE are in-scale only. Validation
is leakage-safe (grouped by source article; entire corpora held out) and every
comparison ships with a paired-bootstrap CI.

## Locked decisions

- **Open difficulty axis, not licensed Lexile.** Free formulas + rank/percentile
  harmonization; the axis only *merges* corpora — human labels stay the target.
- **No paid gold benchmark.** Primary claim = cross-corpus generalization on
  prose (held-out existing human-labeled corpora). Special formats are a $0 pilot.
- **Backbone `roberta-base`** on the modern stack; deberta-v3 requires
  `transformers<4.47` in a separate env.
