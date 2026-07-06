"""External unlabeled pool (Phase 4, first half): fetch diverse free text, window
it to excerpt size, and select for DIVERSITY -- the deliberate inverse of the
winner's nearest-neighbour-to-CLEAR curation, which is what makes the pseudo-
labeled data widen the distribution instead of narrowing it.

Fetch uses HuggingFace `datasets` in streaming mode (bounded by --limit, so no
multi-GB download), imported lazily. Selection is corpus-agnostic and free.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import filter_corpus, window_text
from .schema import Record, coerce, records_to_frame
from .utils import get_logger

log = get_logger("external")

# source -> (hf_dataset_id, config, text_field, domain). Stream + take N.
SOURCES: dict[str, tuple] = {
    "wiki_simple": ("wikimedia/wikipedia", "20231101.simple", "text", "encyclopedic"),
    "wiki_en":     ("wikimedia/wikipedia", "20231101.en",     "text", "encyclopedic"),
    "gutenberg":   ("sedthh/gutenberg_english", None,         "TEXT", "literary"),
}


def fetch_texts(source: str, limit: int) -> list[str]:
    """Stream up to ``limit`` documents' text from a source (lazy `datasets`)."""
    from datasets import load_dataset

    if source not in SOURCES:
        raise KeyError(f"unknown source '{source}'. Known: {sorted(SOURCES)}")
    hf_id, config, field, _ = SOURCES[source]
    ds = load_dataset(hf_id, config, split="train", streaming=True)
    out: list[str] = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        txt = ex.get(field) or ""
        if txt:
            out.append(txt)
    log.info("[%s] fetched %d documents", source, len(out))
    return out


def match_pairs(simple_docs, en_docs, *, n_pairs: int, min_words: int = 80):
    """Title-join two (title, text) iterables into simple<->en article pairs.

    Pure function (no `datasets` dependency) so the join logic is unit-testable;
    build_wikipair_raw feeds it the streamed Wikipedia configs. Skips docs
    shorter than min_words on either side (stubs/disambiguation)."""
    simple_by_title: dict[str, str] = {}
    for title, text in simple_docs:
        t = (title or "").strip()
        if t and text and len(text.split()) >= min_words:
            simple_by_title.setdefault(t, text)
    out = []
    for title, text in en_docs:
        if len(out) >= n_pairs:
            break
        t = (title or "").strip()
        s = simple_by_title.get(t)
        if s and text and len(text.split()) >= min_words:
            out.append({"title": t, "simple_text": s, "en_text": text})
            del simple_by_title[t]          # one pair per title
    log.info("match_pairs: %d pairs (simple pool %d titles)", len(out), len(simple_by_title) + len(out))
    return out


def build_wikipair_raw(dest, *, n_pairs: int = 20000, max_simple: int = 200_000,
                       max_en_scan: int = 2_000_000, min_words: int = 80):
    """Stream wikimedia/wikipedia simple+en (verified configs) and write a raw
    CSV of title-joined article pairs: title, simple_text, en_text.

    ~3.8% of en articles have a simple twin, so finding n_pairs scans roughly
    n_pairs/0.038 en articles; max_en_scan caps the walk."""
    import pandas as pd
    from datasets import load_dataset

    def stream(config, limit):
        ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            yield ex.get("title"), ex.get("text")

    pairs = match_pairs(stream("20231101.simple", max_simple),
                        stream("20231101.en", max_en_scan),
                        n_pairs=n_pairs, min_words=min_words)
    df = pd.DataFrame(pairs)
    df.to_csv(dest, index=False)
    log.info("wikipair raw: %d pairs -> %s", len(df), dest)
    return dest


def difficulty_proxy(text: str) -> float:
    """Cheap, dependency-free readability proxy for *stratification only* (never a
    label): a Flesch-Kincaid-flavoured blend of sentence length and word length."""
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    n_sent = max(len(re.findall(r"[.!?]+", text)), 1)
    words_per_sent = len(words) / n_sent
    chars_per_word = sum(len(w) for w in words) / len(words)
    return 0.39 * words_per_sent + 5.8 * chars_per_word


def build_external_pool(sources: list[str], *, per_source_docs: int,
                        min_tokens: int = 50, max_tokens: int = 320,
                        max_chunks_per_doc: int = 5) -> pd.DataFrame:
    """Fetch -> window -> schema for each source. Rows are unlabeled (native_label
    NaN, split 'unlabeled'); pseudo-labeling fills them in later.

    Keeps at most ``max_chunks_per_doc`` evenly-spaced chunks per document, so a
    single long book (Gutenberg windows into hundreds of chunks) contributes a few
    varied excerpts instead of hundreds of correlated ones -- this bounds memory
    AND boosts diversity (correlated same-author chunks are the opposite of it)."""
    records: list[Record] = []
    for source in sources:
        domain = SOURCES[source][3]
        for di, doc in enumerate(fetch_texts(source, per_source_docs)):
            chunks = window_text(doc, target_words=180, max_words=max_tokens, min_words=min_tokens)
            if len(chunks) > max_chunks_per_doc:
                keep = sorted(set(np.linspace(0, len(chunks) - 1, max_chunks_per_doc).round().astype(int)))
                chunks = [chunks[k] for k in keep]
            for ci, chunk in enumerate(chunks):
                records.append(Record(id=f"{source}:{di}:{ci}", text=chunk, corpus=source,
                                      native_scale="unlabeled", format_type="prose",
                                      domain=domain, language="en", split="unlabeled"))
    df = coerce(records_to_frame(records))
    df = filter_corpus(df, min_tokens=min_tokens, max_tokens=max_tokens)
    log.info("external pool: %d chunks across %s (<=%d per doc)", len(df), sources, max_chunks_per_doc)
    return df


def select_diverse(df: pd.DataFrame, *, n_total: int, n_bins: int = 10,
                   seed: int = 42) -> pd.DataFrame:
    """Stratified-for-diversity sample: bucket by (corpus x difficulty-proxy bin)
    and draw evenly across buckets, so the kept pool spans the full difficulty
    range and every source -- not the slice nearest to CLEAR."""
    if len(df) <= n_total:
        return df.reset_index(drop=True)
    work = df.copy()
    work["_proxy"] = work["text"].map(difficulty_proxy)
    # rank-based bins are robust to the proxy's arbitrary scale
    work["_bin"] = pd.qcut(work["_proxy"].rank(method="first"), q=n_bins, labels=False)
    rng = np.random.default_rng(seed)
    groups = list(work.groupby(["corpus", "_bin"]))
    per = max(n_total // max(len(groups), 1), 1)
    picks = []
    for _, g in groups:
        take = min(per, len(g))
        picks.append(g.sample(n=take, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks, ignore_index=True)
    if len(out) > n_total:                      # trim overshoot uniformly
        out = out.sample(n=n_total, random_state=seed)
    log.info("select_diverse: %d -> %d across %d buckets", len(df), len(out), len(groups))
    return out.drop(columns=["_proxy", "_bin"]).reset_index(drop=True)
