"""Config loading. Plain YAML -> nested dataclasses, with CLI dotted overrides.

No hard Hydra dependency (one fewer install for the team), but the structure is
Hydra-compatible: every experimental variant is a config value, never a code
edit, so the Phase-8 ablation matrix is just a set of YAML overrides.

    cfg = load_config("configs/default.yaml", overrides=["train.epochs=5"])

Note: model/train fields drive the C++ model core under /src; YAML is language-
agnostic, so the same config file configures both the Python prep/eval and the
C++ trainer (which reads the same config path).
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class PathsConfig:
    raw: str = "data"            # raw downloads only
    artifacts: str = "artifacts" # derived tables / predictions / embeddings
    runs: str = "runs"


@dataclass
class DataConfig:
    corpora: list[str] = field(default_factory=lambda: ["clear"])
    unified_table: str = "artifacts/corpus.csv"
    min_tokens: int = 3
    max_tokens: int = 600
    wikipair_pairs: int = 20000   # simple<->en article pairs to build (corpus "wikipair")


@dataclass
class SplitConfig:
    holdout_corpora: list[str] = field(default_factory=list)
    holdout_formats: list[str] = field(default_factory=list)
    group_by: str = "corpus"
    n_folds: int = 5
    val_fraction: float = 0.1
    seed: int = 42


@dataclass
class ModelConfig:
    # Consumed by the C++ trainer (/src). Kept here so prep/eval and the model
    # share one config surface.
    backbone: str = "roberta-base"   # deberta-v3 NaNs on transformers>=~4.47 (use <4.47 for it)
    max_length: int = 512
    dropout: float = 0.1
    # Run-2 ablation (2026-07, 3 seeds, paired bootstrap on ood_corpus): the
    # unmasked pairwise head measured net-NEGATIVE for transfer (removing it
    # +0.003, CI excl. 0) -> default off. The loss is now masked to same-source
    # pairs; re-test deliberately before re-enabling.
    use_pairwise_head: bool = False
    # Source offset also measured negative under SINGLE-gold training (removing
    # it +0.008, CI excl. 0) -- but it exists for MULTI-gold scales, the next
    # default experiment (LOCO). Keep on for multi-gold; set false for
    # single-gold cross-corpus runs.
    use_source_offset: bool = True
    # Concat the standardized formula proxy to the pooled representation before
    # the head. Motivated by the per-corpus formula floors (2026-07-03): the
    # proxy alone scored 0.79 on CEFR (above every model) and 0.97 within-article
    # on OneStop -- complementary signal the encoder misses. Off by default until
    # the LOCO+hybrid run measures it.
    use_formula_feature: bool = False
    peft: bool = False


@dataclass
class TrainConfig:
    stage: str = "finetune"          # "pretrain" (pseudo) | "finetune" (gold)
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    pointwise_weight: float = 1.0
    pairwise_weight: float = 1.0
    confidence_weighting: bool = True
    two_stage: bool = False          # strict pretrain-on-pseudo -> finetune-on-gold
    pretrain_lr: float = 1.0e-5      # Stage A (pseudo) learning rate, lower than finetune
    pretrain_epochs: int = 1
    seed: int = 42


@dataclass
class EvalConfig:
    se_col: str = "BT s.e."
    target_col: str = "BT Easiness"
    formula_cols: list[str] = field(default_factory=lambda: [
        "Flesch-Reading-Ease", "Flesch-Kincaid-Grade-Level",
        "New Dale-Chall Readability Formula", "CAREC",
    ])
    solution_cols: list[str] = field(default_factory=lambda: [
        "firstPlace_pred", "secondPlace_pred", "thirdPlace_pred",
    ])
    n_bootstrap: int = 1000
    seed: int = 42


@dataclass
class ExternalConfig:
    sources: list[str] = field(default_factory=lambda: ["wiki_simple", "wiki_en", "gutenberg"])
    per_source_docs: int = 5000        # streamed per source (bounded; raise on the H100)
    n_total: int = 200_000             # diverse-selected pool size
    n_bins: int = 10
    max_chunks_per_doc: int = 5         # cap per document (bounds memory + boosts diversity)
    pool_table: str = "artifacts/external_pool.csv"


@dataclass
class TeacherConfig:
    backbone: str = "roberta-base"   # deberta-v3 NaNs on transformers>=~4.47 (use <4.47 for it)
    n_teachers: int = 3                # ensemble for disagreement filtering
    target_col: str = "native_label"   # teacher predicts CLEAR BT (SE-filter in BT units)
    dir: str = "models/teachers"


@dataclass
class PseudoLabelConfig:
    k_se: float = 1.0                  # keep |pred - neighbour| <= k_se * neighbour s.e.
    max_std: float | None = None       # disagreement gate (None -> median split)
    dedup_cosine: float = 0.05         # drop pool chunks within this cosine dist of a gold passage
    extrapolate: bool = True           # extrapolate the axis beyond CLEAR's range (vs clamp)
    embed_backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    out_table: str = "artifacts/pseudo_labeled.csv"
    train_pool_table: str = "artifacts/train_pool.csv"  # gold + pseudo, ready for the student


@dataclass
class Config:
    experiment: str = "default"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    external: ExternalConfig = field(default_factory=ExternalConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    pseudo: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    # get_type_hints resolves string annotations (we use `from __future__ import
    # annotations`) back to real types, so nested config dataclasses are detected.
    hints = get_type_hints(cls)
    for key, value in (data or {}).items():
        if key not in hints:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        ftype = hints[key]
        kwargs[key] = _from_dict(ftype, value) if is_dataclass(ftype) and isinstance(value, dict) else value
    return cls(**kwargs)


def _apply_override(cfg: Config, dotted: str) -> None:
    key, sep, raw = dotted.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got '{dotted}'")
    value = yaml.safe_load(raw)
    obj: Any = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(f"unknown override target '{key}'")
    setattr(obj, parts[-1], value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _from_dict(Config, raw)
    for ov in overrides or []:
        _apply_override(cfg, ov)
    return cfg
