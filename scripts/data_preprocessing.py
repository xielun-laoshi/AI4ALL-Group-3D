#!/usr/bin/env python
"""Reproduce the data preprocessing end to end (framework Phases 1-2).

One consolidated entry point for the four steps that used to be separate
scripts: download -> build unified schema -> harmonize onto the open axis ->
assign walled-off / cross-corpus splits.

    # full prep with CLEAR only:
    python scripts/data_preprocessing.py

    # hold out a corpus and a format as the cross-corpus / cross-format gold set:
    python scripts/data_preprocessing.py --holdout-corpora onestop --holdout-formats poetry

Raw downloads land in data/ (raw docs only). The derived unified table is written
to artifacts/ (never committed). CLEAR is public and free, so a clean checkout
reproduces everything without any gated download.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# scripts/ is on sys.path[0] when this file is run directly, so `readability`
# (the support lib at scripts/readability/) imports without any install step.
from readability.config import load_config
from readability.data import assign_splits, filter_corpus, find_clear, harmonize, load_corpus
from readability.schema import CANONICAL_COLUMNS, coerce, validate, write_table
from readability.utils import artifacts_dir, data_dir, get_logger, seed_everything

log = get_logger("preprocess")

# Public, free raw sources. Each entry: {"url": <url or None>, "file"|"dir":
# <local target under data/>, "archive": "zip" (optional)}.  url=None => a
# manual/licensed source (documented, not auto-fetched).
RAW_SOURCES = {
    "clear": {"url": "https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/main/CLEAR_corpus_final.xlsx",
              "file": "CLEAR.xlsx"},   # upstream ships only .xlsx; read_clear handles it
    "onestop": {"url": "https://github.com/nishkalavallabhi/OneStopEnglishCorpus/archive/refs/heads/master.zip",
                "dir": "onestop", "archive": "zip"},
    "cefr": {"url": "https://raw.githubusercontent.com/AMontgomerie/CEFR-English-Level-Predictor/main/data/cefr_leveled_texts.csv",
             "file": "cefr.csv"},
    # built (not downloaded): title-join of wikimedia/wikipedia simple x en via
    # `datasets` streaming -- see readability.external.build_wikipair_raw.
    "wikipair": {"url": None, "file": "wikipair.csv", "builder": "wikipair"},
    "newsela": {"url": None, "dir": "newsela"},      # newsela.com/data (free, on request)
    "weebit": {"url": None, "dir": "weebit"},        # request from authors
    "gutenberg_poetry": {"url": None, "dir": "gutenberg_poetry"},  # Project Gutenberg
}


def local_raw_path(corpus: str) -> Path:
    """Where this corpus's raw data lives under data/ (a file or a directory)."""
    if corpus == "clear":
        return find_clear(data_dir())   # prefer an existing CLEAR.csv over the downloaded xlsx
    spec = RAW_SOURCES[corpus]
    return data_dir() / (spec["file"] if "file" in spec else spec["dir"])


def ensure_raw(corpus: str, force: bool, cfg=None) -> Path:
    """Make sure the raw data is present under data/, downloading (or building)
    if missing. Returns a file path (single-file corpora) or a directory."""
    spec = RAW_SOURCES.get(corpus)
    if spec is None:
        raise SystemExit(f"[{corpus}] unknown corpus (no source configured).")
    dest = local_raw_path(corpus)
    present = dest.exists() and (any(dest.iterdir()) if dest.is_dir() else True)
    if present and not force:
        log.info("[%s] using existing %s", corpus, dest)
        return dest
    if spec.get("builder") == "wikipair":
        from readability.external import build_wikipair_raw

        n_pairs = cfg.data.wikipair_pairs if cfg is not None else 20000
        log.info("[wikipair] building %d simple<->en pairs (streams wikimedia/wikipedia; "
                 "scans ~%dk en articles)", n_pairs, int(n_pairs / 0.038 / 1000))
        data_dir().mkdir(parents=True, exist_ok=True)
        return Path(build_wikipair_raw(dest, n_pairs=n_pairs))
    if spec.get("url") is None:
        raise SystemExit(
            f"[{corpus}] no public download configured -- fetch it manually into "
            f"{dest} (see RAW_SOURCES notes), then re-run with --skip-download.")
    data_dir().mkdir(parents=True, exist_ok=True)
    if spec.get("archive") == "zip":
        zip_path = data_dir() / f"{corpus}.zip"
        log.info("[%s] downloading archive -> %s", corpus, zip_path)
        urllib.request.urlretrieve(spec["url"], zip_path)
        log.info("[%s] extracting -> %s", corpus, dest)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        zip_path.unlink(missing_ok=True)
        return dest
    log.info("[%s] downloading -> %s", corpus, dest)
    urllib.request.urlretrieve(spec["url"], dest)
    log.info("[%s] downloaded %d bytes", corpus, dest.stat().st_size)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--corpora", nargs="+", default=None, help="override config corpora")
    ap.add_argument("--holdout-corpora", nargs="*", default=None)
    ap.add_argument("--holdout-formats", nargs="*", default=None)
    ap.add_argument("--harmonize-method", default="percentile")
    ap.add_argument("--out", default=None, help="output table (default: artifacts/corpus.csv)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.splits.seed)
    corpora = args.corpora or cfg.data.corpora
    holdout_corpora = args.holdout_corpora if args.holdout_corpora is not None else cfg.splits.holdout_corpora
    holdout_formats = args.holdout_formats if args.holdout_formats is not None else cfg.splits.holdout_formats
    out_path = args.out or str(artifacts_dir() / "corpus.csv")

    # 1. download + 2. load into the unified schema
    frames = []
    for name in corpora:
        raw = local_raw_path(name) if args.skip_download else ensure_raw(name, args.force_download, cfg)
        frames.append(load_corpus(name, raw))
    df = coerce(pd.concat(frames, ignore_index=True))

    # 2b. QC filters: empties, length band, language, exact-dedup within corpus
    df = filter_corpus(df, min_tokens=cfg.data.min_tokens, max_tokens=cfg.data.max_tokens)

    # 3. harmonize onto the open [0, 1] axis
    df = harmonize(df, method=args.harmonize_method)

    # 4. assign walled-off / cross-corpus splits
    df = assign_splits(df, holdout_corpora=holdout_corpora,
                       holdout_formats=holdout_formats,
                       val_fraction=cfg.splits.val_fraction, seed=cfg.splits.seed)

    for msg in validate(df, strict=False):
        log.warning("schema: %s", msg)
    written = write_table(df.reindex(columns=CANONICAL_COLUMNS), out_path)
    log.info("wrote %d rows -> %s | corpora=%s", len(df), written, df["corpus"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
