"""Data layer: corpus loaders -> unified schema, open-axis harmonization, and
gold-walled / cross-corpus splitting (framework Phases 1-2).

Merged into one module because these are the three "data preprocessing" steps
the single ``scripts/data_preprocessing.py`` entry point chains together.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .schema import CANONICAL_COLUMNS, Record, coerce, records_to_frame
from .utils import get_logger

log = get_logger("data")


# --------------------------------------------------------------------------- #
# 1. Loaders: each maps one raw source into the unified schema.               #
# --------------------------------------------------------------------------- #
# CLEAR ships upstream as CLEAR_corpus_final.xlsx (scrosseye/CLEAR-Corpus, main)
# with drifted column names and NO winner-prediction columns; the AI4ALL team's
# CSV uses the canonical names. These aliases normalize whichever file is present.
CLEAR_COLUMN_ALIASES = {
    "BT_easiness": "BT Easiness",
    "s.e.": "BT s.e.",
    "Categ": "Category",
}


def find_clear(data_dir_path: str | Path) -> Path:
    """Locate the local CLEAR file, preferring an existing CSV (e.g. a manually
    materialized one) over the downloaded xlsx."""
    d = Path(data_dir_path)
    for name in ("CLEAR.csv", "CLEAR.xlsx"):
        if (d / name).exists():
            return d / name
    return d / "CLEAR.xlsx"


def read_clear(path: str | Path) -> pd.DataFrame:
    """Read CLEAR from .xlsx or .csv and normalize drifted column names
    (BT_easiness/s.e./Categ -> BT Easiness/BT s.e./Category). xlsx needs openpyxl."""
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    rename = {a: c for a, c in CLEAR_COLUMN_ALIASES.items() if a in df.columns and c not in df.columns}
    if rename:
        df = df.rename(columns=rename)
        log.info("read_clear: aliased columns %s", rename)
    return df


def load_clear(raw_path: str | Path) -> pd.DataFrame:
    """Load the CLEAR corpus (.csv or .xlsx) into the unified schema.

    Keeps the human BT easiness as ``native_label`` and ``BT s.e.`` as
    ``std_error``. Reference columns (formulas, winner preds) stay in the raw
    file -- the eval harness reads them directly from there.
    """
    df = read_clear(raw_path)
    cat = df.get("Category", pd.Series(index=df.index, dtype="object")).astype(str).str.lower()
    domain = cat.map({"lit": "literary", "info": "informational"}).fillna("unknown")
    # CLEAR's ID column parses as float (some rows have no ID); format as a clean
    # int and fall back to a positional id so the few label-less rows stay unique.
    raw_ids = pd.to_numeric(df["ID"], errors="coerce")
    ids = [f"clear:{int(v)}" if pd.notna(v) else f"clear:row{i}"
           for i, v in enumerate(raw_ids)]
    out = pd.DataFrame({
        "id": ids,
        "text": df["Excerpt"].astype(str),
        "corpus": "clear",
        "native_label": pd.to_numeric(df["BT Easiness"], errors="coerce"),
        "native_scale": "clear_bt_easiness",
        "harmonized_difficulty": pd.NA,
        "mapping_method": "none",
        "mapping_confidence": pd.NA,
        "std_error": pd.to_numeric(df["BT s.e."], errors="coerce"),
        "format_type": "prose",
        "domain": domain,
        "language": "en",
        "length_tokens": df["Excerpt"].astype(str).str.split().str.len(),
        "license": df.get("License", pd.Series("unknown", index=df.index)).astype(str),
        "split": "unassigned",
        "is_pseudo": False,
    })
    log.info("loaded CLEAR: %d rows", len(out))
    return coerce(out)


# OneStopEnglish reading levels -> ordinal native_label (higher = harder).
ONESTOP_LEVELS = {"ele": 0, "int": 1, "adv": 2}


def _read_text(path: Path) -> str:
    """Read a text file, tolerating the corpus's mixed encodings."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def window_text(text: str, *, target_words: int = 180, max_words: int = 320,
                min_words: int = 50) -> list[str]:
    """Split a long document into excerpt-sized chunks (~target_words), packing on
    paragraph boundaries so chunks don't cut mid-sentence. Oversized paragraphs
    are split on word count; a too-short trailing chunk is merged back.

    Operates on one document at a time, so it streams to arbitrarily large corpora.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    buf_n = 0
    for p in paras:
        words = p.split()
        if len(words) > max_words:                       # paragraph alone too big
            if buf:
                chunks.append(" ".join(buf)); buf, buf_n = [], 0
            for i in range(0, len(words), target_words):
                chunks.append(" ".join(words[i:i + target_words]))
            continue
        if buf_n and buf_n + len(words) > target_words:  # flush before overflow
            chunks.append(" ".join(buf)); buf, buf_n = [], 0
        buf.append(p); buf_n += len(words)
    if buf:
        chunks.append(" ".join(buf))
    if len(chunks) >= 2 and len(chunks[-1].split()) < min_words:
        # Pop FIRST, then merge. The one-liner `chunks[-2] = ... + chunks.pop()`
        # evaluates the RHS (shrinking the list) before resolving the assignment
        # target: IndexError on 2-chunk docs, and on >=3 chunks it assigned to the
        # wrong slot (losing one chunk and duplicating another).
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + " " + tail             # absorb a short tail
    return chunks


def load_onestop(raw_dir: str | Path) -> pd.DataFrame:
    """Load OneStopEnglish (a tree of *-ele/-int/-adv .txt files) into the unified
    schema. Each article exists at three expert reading levels; we window every
    article into excerpt-sized chunks that inherit its level.

    native_label = ordinal level (0/1/2); std_error is NaN (expert levels carry no
    per-item s.e.). format=prose, domain=informational (re-leveled news).
    """
    raw_dir = Path(raw_dir)
    records: list[Record] = []
    n_files = 0
    for level, label in ONESTOP_LEVELS.items():
        seen: set[str] = set()
        # restrict to the level-separated texts; the repo ships a nested duplicate
        # Int-Txt/Int-Txt copy, so dedup by filename to avoid loading each twice.
        for fp in sorted(raw_dir.glob(f"**/Texts-SeparatedByReadingLevel/**/*-{level}.txt")):
            if fp.name in seen:
                continue
            seen.add(fp.name)
            n_files += 1
            article = fp.stem.rsplit("-", 1)[0]
            for ci, chunk in enumerate(window_text(_read_text(fp))):
                records.append(Record(
                    id=f"onestop:{article}:{level}:{ci}",
                    text=chunk, corpus="onestop",
                    native_label=float(label), native_scale="onestop_level",
                    format_type="prose", domain="informational",
                    language="en", license="CC-BY-SA-4.0",
                ))
    if n_files == 0:
        raise FileNotFoundError(
            f"no OneStopEnglish texts found under {raw_dir}. Expected a "
            f"'Texts-SeparatedByReadingLevel/' tree with *-ele/-int/-adv.txt files.")
    df = records_to_frame(records)
    log.info("loaded OneStopEnglish: %d chunks from %d files", len(df), n_files)
    return coerce(df)


# CEFR proficiency levels -> ordinal native_label (higher = harder).
CEFR_LEVELS = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}


def load_cefr(raw_path: str | Path) -> pd.DataFrame:
    """Load the CEFR Levelled English Texts CSV (columns text,label) into the
    unified schema. A *second* cross-corpus holdout, distinct in source from CLEAR
    and OneStopEnglish, with a finer 6-level target (A1..C2 -> 0..5). Texts are
    windowed to excerpt size; std_error is NaN (no per-item noise)."""
    df = pd.read_csv(raw_path)
    records: list[Record] = []
    for ri, row in enumerate(df.itertuples(index=False)):
        level = str(getattr(row, "label", "")).strip().upper()
        if level not in CEFR_LEVELS:
            continue
        for ci, chunk in enumerate(window_text(str(getattr(row, "text", "")))):
            records.append(Record(id=f"cefr:{level}:{ri}:{ci}", text=chunk, corpus="cefr",
                                  native_label=float(CEFR_LEVELS[level]), native_scale="cefr_level",
                                  format_type="prose", domain="mixed", language="en",
                                  license="research"))
    if not records:
        raise ValueError(f"no CEFR rows parsed from {raw_path} (expected columns text,label)")
    out = records_to_frame(records)
    log.info("loaded CEFR: %d chunks from %d texts", len(out), len(df))
    return coerce(out)


def _stub_loader(name: str) -> Callable[[str | Path], pd.DataFrame]:
    def _loader(raw_path: str | Path) -> pd.DataFrame:
        raise NotImplementedError(
            f"loader for '{name}' not implemented yet. Add it here once the raw "
            f"data is downloaded; it must return the canonical schema with "
            f"native_label on this corpus's own scale."
        )
    _loader.__name__ = f"load_{name}"
    return _loader


# Phase-1 registry. Tier-1 = human-labeled backbone; Tier-2 = grade/Lexile
# anchors; Tier-3 = unlabeled breadth incl. the special formats.
REGISTRY: dict[str, Callable[[str | Path], pd.DataFrame]] = {
    "clear": load_clear,
    "onestop": load_onestop,
    "cefr": load_cefr,
    "newsela": _stub_loader("newsela"),
    "weebit": _stub_loader("weebit"),
    "wiki_simple": _stub_loader("wiki_simple"),
    "gutenberg_poetry": _stub_loader("gutenberg_poetry"),
}


def load_corpus(name: str, raw_path: str | Path) -> pd.DataFrame:
    if name not in REGISTRY:
        raise KeyError(f"unknown corpus '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](raw_path)


def filter_corpus(df: pd.DataFrame, *, min_tokens: int = 3, max_tokens: int = 600,
                  languages: tuple[str, ...] = ("en",)) -> pd.DataFrame:
    """Phase-1 QC filters: drop empties, enforce a comparable length band so the
    model can't learn length instead of difficulty, keep target languages, and
    drop exact-duplicate texts within a corpus. Dedup is per (corpus, text), so
    parallel levels (same article re-leveled) are preserved. Vectorized -> scales.
    """
    n0 = len(df)
    df = df[df["text"].astype(str).str.strip().ne("")]
    lt = pd.to_numeric(df["length_tokens"], errors="coerce")
    df = df[(lt >= min_tokens) & (lt <= max_tokens)]
    df = df[df["language"].isin(languages)]
    df = df.drop_duplicates(subset=["corpus", "text"], keep="first")
    log.info("filter_corpus: %d -> %d rows (min=%d max=%d)", n0, len(df), min_tokens, max_tokens)
    return df.reset_index(drop=True)


def _norm_text(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def dedup_against(df: pd.DataFrame, reference: pd.DataFrame, *, key: str = "text") -> pd.DataFrame:
    """Drop rows of ``df`` whose normalized text exactly matches any reference text.

    Integrity guard for pseudo-labeling: CLEAR is sourced from Gutenberg/Wikipedia --
    the same pools we sample for the external set -- so external chunks can be
    identical to gold/holdout passages. This removes those before they are pseudo-
    labeled; the embedding SE-filter additionally catches near-duplicates."""
    ref = set(reference[key].map(_norm_text))
    keep = ~df[key].map(_norm_text).isin(ref)
    n_drop = int((~keep).sum())
    if n_drop:
        log.info("dedup_against: dropped %d/%d rows matching reference text", n_drop, len(df))
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Harmonization: native labels -> open [0, 1] difficulty axis.            #
# --------------------------------------------------------------------------- #
# Guardrail: this axis only *merges* corpora; supervision/eval stay on the human
# label. CLEAR's BT easiness is higher=EASIER, so its polarity is inverted.
POLARITY = {"clear": False, "onestop": True, "cefr": True}   # higher native label == harder?


def percentile_within_corpus(df: pd.DataFrame, *, label_col: str = "native_label",
                             corpus_col: str = "corpus",
                             polarity: dict[str, bool] | bool = True) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for corpus, g in df.groupby(corpus_col):
        vals = pd.to_numeric(g[label_col], errors="coerce")
        harder_up = polarity.get(str(corpus), True) if isinstance(polarity, dict) else polarity
        ranks = vals.rank(pct=True)
        out.loc[g.index] = ranks if harder_up else (1.0 - ranks)
    return out


def harmonize(df: pd.DataFrame, *, method: str = "percentile",
              polarity: dict[str, bool] | bool = POLARITY) -> pd.DataFrame:
    out = df.copy()
    if method == "percentile":
        out["harmonized_difficulty"] = percentile_within_corpus(out, polarity=polarity)
        out["mapping_method"] = "percentile_within_corpus"
        out["mapping_confidence"] = np.where(out["harmonized_difficulty"].notna(), 1.0, np.nan)
    elif method in {"band_table", "isotonic"}:
        raise NotImplementedError(
            f"harmonization method '{method}' is a Phase-1 stub; implement once "
            f"the Tier-2 grade/CEFR anchor tables are available."
        )
    else:
        raise ValueError(f"unknown harmonization method '{method}'")
    return out


# --------------------------------------------------------------------------- #
# 3. Splits: wall off gold by holding out entire corpora / formats.          #
# --------------------------------------------------------------------------- #
def assign_splits(df: pd.DataFrame, *, holdout_corpora: list[str] | None = None,
                  holdout_formats: list[str] | None = None,
                  val_fraction: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Assign ``split``. Held-out corpora/formats become the cross-corpus /
    cross-format gold test; the rest splits into train/val. Never random across a
    source -- that's the fix for the legacy notebooks' in-distribution KFold."""
    holdout_corpora = set(holdout_corpora or [])
    holdout_formats = set(holdout_formats or [])
    out = df.copy()
    rng = np.random.default_rng(seed)

    has_label = pd.to_numeric(out["native_label"], errors="coerce").notna().to_numpy()
    split = np.where(has_label, "train", "unlabeled").astype(object)
    split[out["corpus"].isin(holdout_corpora).to_numpy() & has_label] = "ood_corpus"
    split[out["format_type"].isin(holdout_formats).to_numpy() & has_label] = "ood_format"

    train_idx = np.where(split == "train")[0]
    n_val = int(len(train_idx) * val_fraction)
    if n_val > 0:
        split[rng.choice(train_idx, size=n_val, replace=False)] = "val"

    out["split"] = split
    log.info("split assignment: %s", out["split"].value_counts().to_dict())
    return out


def derive_group_id(df: pd.DataFrame) -> pd.Series:
    """Leakage-safe grouping key derived from the row id.

    Keeps every chunk AND every reading-level of one source article together so a
    near-duplicate can't straddle the train/val boundary. ids are structured as
    ``corpus:article:level:chunk`` (OneStopEnglish, 4 parts) or ``corpus:n``
    (CLEAR, 2 parts); we collapse the former to ``corpus:article`` and leave flat
    ids as their own group.
    """
    def base(rid: str) -> str:
        parts = str(rid).split(":")
        return ":".join(parts[:2]) if len(parts) >= 4 else str(rid)
    return df["id"].map(base)


def cv_folds(df: pd.DataFrame, *, group_by: str = "auto", n_folds: int = 5,
             pool_splits: tuple[str, ...] = ("train", "val")):
    """Yield leakage-safe (train_idx, val_idx) cross-validation folds.

    ``group_by``:
      "auto"   -> group by source article/passage (derive_group_id). The always-on
                  integrity fix: OneStopEnglish's three reading-levels and every
                  windowed chunk of one article stay inside a single fold.
      "corpus" -> hold out an entire corpus per fold (cross-corpus transfer CV;
                  needs >= n_folds corpora in the pool).
      <column> -> group by that schema column.

    Operates on the {train, val} pool only -- the gold ood_* holdout is never part
    of CV. Raises if there are fewer groups than folds (the split would be invalid).
    """
    from sklearn.model_selection import GroupKFold

    pool = df[df["split"].isin(set(pool_splits))]
    if group_by == "auto":
        groups = derive_group_id(pool).to_numpy()
    elif group_by in pool.columns:
        groups = pool[group_by].astype(str).to_numpy()
    else:
        raise ValueError(f"group_by must be 'auto', 'corpus', or a column name; got '{group_by}'")

    n_groups = len(set(groups))
    if n_groups < n_folds:
        raise ValueError(
            f"only {n_groups} group(s) for group_by='{group_by}' but n_folds={n_folds}; "
            f"use fewer folds, a finer group_by, or leave-one-group-out.")
    gkf = GroupKFold(n_splits=n_folds)
    for tr, va in gkf.split(np.zeros(len(pool)), groups=groups):
        yield pool.index[tr].to_numpy(), pool.index[va].to_numpy()
