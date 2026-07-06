"""Smoke tests for the Phase-0 scaffold: schema contract, harmonization
polarity, metrics, and the reference bracket. Run: ``pytest``."""

import numpy as np
import pandas as pd

from readability.schema import CANONICAL_COLUMNS, Record, coerce, records_to_frame, validate
from readability.data import percentile_within_corpus, POLARITY, derive_group_id, cv_folds, dedup_against
from readability.utils import seed_everything
from readability.evaluation import spearman, pairwise_accuracy, rmse, mean_predictor_rmse
from readability.external import difficulty_proxy, select_diverse
from readability.pseudolabel import clear_bt_to_axis, generate_pseudo_labels
from readability.ablation import aggregate, paired_bootstrap_diff


def _toy():
    recs = [
        Record(id="clear:1", text="The cat sat.", corpus="clear", native_label=1.5,
               native_scale="clear_bt_easiness", std_error=0.4),
        Record(id="clear:2", text="Notwithstanding the epistemic quandary therein.",
               corpus="clear", native_label=-2.0, native_scale="clear_bt_easiness", std_error=0.5),
        Record(id="onestop:1", text="A simple sentence here.", corpus="onestop",
               native_label=2.0, native_scale="grade", std_error=None),
    ]
    return records_to_frame(recs)


def test_schema_columns_and_validation():
    df = _toy()
    assert list(df.columns) == CANONICAL_COLUMNS
    assert validate(df, strict=False) == []  # no violations on a clean toy frame


def test_length_tokens_autofilled():
    assert (_toy()["length_tokens"] > 0).all()


def test_percentile_polarity_inverts_clear():
    df = _toy()
    # CLEAR easiness: higher == easier, so POLARITY inverts -> the very negative
    # excerpt is the hardest (~1.0).
    hd = percentile_within_corpus(df, polarity=POLARITY)
    clear = df["corpus"] == "clear"
    hardest = df.loc[clear, "native_label"].idxmin()
    easiest = df.loc[clear, "native_label"].idxmax()
    assert hd[hardest] > hd[easiest]


def test_metrics_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.2, 3.8])
    assert spearman(y, p) == 1.0
    assert pairwise_accuracy(y, p) == 1.0
    assert rmse(y, y) == 0.0


def test_mean_predictor_rmse_equals_std():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(mean_predictor_rmse(y) - y.std(ddof=0)) < 1e-9


def test_rank_rmse_is_scale_invariant():
    from readability.evaluation import rank_rmse
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = 10 * y + 100                      # perfect ranking, wildly different scale
    assert rank_rmse(y, p) < 1e-9         # scale-free: sees the perfect ranking
    assert rmse(y, p) > 100               # raw RMSE is huge and would mislead cross-corpus
    assert rank_rmse(y, y[::-1].copy()) > rank_rmse(y, p)   # reversed ranking is worse


def test_derive_group_id_collapses_onestop_levels():
    df = pd.DataFrame({"id": ["onestop:Amazon:ele:0", "onestop:Amazon:adv:1", "clear:42"]})
    g = derive_group_id(df).tolist()
    # all reading-levels / chunks of one article share a group; flat ids stand alone
    assert g[0] == g[1] == "onestop:Amazon"
    assert g[2] == "clear:42"


def test_cv_folds_never_leak_a_group():
    # 10 articles x 3 reading-levels = 30 rows, 10 groups, 5 folds.
    rows = [{"id": f"onestop:art{a}:{lvl}:0", "split": "train"}
            for a in range(10) for lvl in ("ele", "int", "adv")]
    df = pd.DataFrame(rows)
    validated = []
    for tr, va in cv_folds(df, group_by="auto", n_folds=5):
        gtr = set(derive_group_id(df.loc[tr]))
        gva = set(derive_group_id(df.loc[va]))
        assert gtr.isdisjoint(gva)                       # no article on both sides
        validated.append(gva)
    # every article is validated exactly once across the folds
    assert set().union(*validated) == {f"onestop:art{a}" for a in range(10)}


def test_read_clear_aliases_drifted_columns(tmp_path):
    from readability.data import read_clear
    csv = tmp_path / "clear.csv"
    csv.write_text("ID,Excerpt,BT_easiness,s.e.,Categ\n1,hello world,-0.5,0.4,Lit\n", encoding="utf-8")
    df = read_clear(csv)  # upstream names should be aliased to canonical
    assert {"BT Easiness", "BT s.e.", "Category"} <= set(df.columns)
    assert df["BT Easiness"].iloc[0] == -0.5


def test_pairwise_loss_masks_cross_source_pairs():
    import torch
    from readability.training import _pairwise_loss
    # Within each source the ordering is CORRECT; every wrong ordering is a
    # cross-source pair. Masked loss must be 0; unmasked must be positive.
    scores = torch.tensor([1.0, 2.0, -5.0, -4.0])
    targets = torch.tensor([0.1, 0.9, 0.2, 0.8])
    sources = torch.tensor([0, 0, 1, 1])
    assert float(_pairwise_loss(scores, targets, sources)) == 0.0
    assert float(_pairwise_loss(scores, targets, None)) > 0.0


def test_window_text_two_chunk_short_tail_does_not_crash():
    from readability.data import window_text
    # 180-word para + 20-word tail para -> 2 chunks, tail < min_words.
    # The old one-liner merge popped before resolving chunks[-2] -> IndexError.
    doc = ("alpha " * 180).strip() + "\n\n" + ("tail " * 20).strip()
    out = window_text(doc)
    assert len(out) == 1
    assert "alpha" in out[0] and "tail" in out[0]        # merged, nothing lost


def test_window_text_short_tail_merge_preserves_all_chunks():
    from readability.data import window_text
    # 3 paras -> 3 chunks with a short tail. The old merge silently LOST the
    # first chunk and duplicated the second; the fix merges tail into the last.
    doc = ("alpha " * 180).strip() + "\n\n" + ("bravo " * 180).strip() + "\n\n" + ("tail " * 20).strip()
    out = window_text(doc)
    assert len(out) == 2
    assert any("alpha" in c for c in out)                # first chunk survives
    assert "tail" in out[-1] and "bravo" in out[-1]      # tail merged into last
    assert sum(1 for c in out if "bravo" in c) == 1      # no duplication


def test_onestop_decomposition_within_vs_across():
    from analyze_transfer import onestop_decomposition
    # artA: levels ordered correctly by pred; artB: levels INVERTED.
    rows = []
    for art, flip in (("artA", False), ("artB", True)):
        for lvl in (0, 1, 2):
            pred = (2 - lvl if flip else lvl) * 0.3 + 0.1
            rows.append({"id": f"onestop:{art}:{lvl}:0", "corpus": "onestop",
                         "native_label": float(lvl), "pred": pred})
    out = onestop_decomposition(pd.DataFrame(rows))
    assert out["n_articles"] == 2
    assert out["n_within_pairs"] == 6                  # 3 level-pairs per article
    assert abs(out["within_article_acc"] - 0.5) < 1e-9  # one article right, one inverted


# --- Wiki-pairs construct experiment ----------------------------------------- #
def test_match_pairs_title_join():
    from readability.external import match_pairs
    long = lambda s: (s + " word") * 100
    simple = [("Cat", long("simple cat")), ("Dog", long("simple dog")), ("Stub", "too short")]
    en = [("Cat", long("complex cat")), ("Bird", long("no twin")), ("Stub", long("long enough")),
          ("Dog", long("complex dog"))]
    out = match_pairs(simple, en, n_pairs=10, min_words=50)
    assert {p["title"] for p in out} == {"Cat", "Dog"}   # Stub too short, Bird unmatched


def test_load_wikipair_requires_both_sides(tmp_path):
    from readability.data import load_wikipair
    body = ("word " * 120).strip()
    csv = tmp_path / "wikipair.csv"
    pd.DataFrame({"title": ["A", "B"],
                  "simple_text": [body, ""],          # B has no simple side -> dropped
                  "en_text": [body + " extra", body]}).to_csv(csv, index=False)
    out = load_wikipair(csv)
    assert set(out["corpus"]) == {"wikipair"}
    parts = out["id"].str.split(":")
    assert set(parts.str[1]) == {"0"}                   # only pair A survives
    assert set(parts.str[2]) == {"simple", "en"}        # both sides present
    assert set(out["native_label"]) == {0.0, 1.0}


def test_pairing_group_ids_groups_pairs_only():
    from readability.data import pairing_group_ids
    df = pd.DataFrame({"id": ["wikipair:7:simple:0", "wikipair:7:en:0",
                              "wikipair:8:en:0", "clear:1", "clear:2"],
                       "corpus": ["wikipair"] * 3 + ["clear"] * 2})
    g = pairing_group_ids(df)
    assert g[0] == g[1]                                 # same article -> same group
    assert g[2] not in (g[0], g[3], g[4])
    assert len({g[3], g[4]} & {g[0], g[1], g[2]}) == 0 and g[3] != g[4]  # clear rows unique


def test_harmonize_zeroes_pair_confidence_and_val_excludes_pairs():
    from readability.data import harmonize, assign_splits
    rows = [{"id": f"clear:{i}", "text": "t", "corpus": "clear", "native_label": float(i),
             "format_type": "prose"} for i in range(20)]
    rows += [{"id": f"wikipair:{i}:{lvl}:0", "text": "t", "corpus": "wikipair",
              "native_label": lab, "format_type": "prose"}
             for i in range(10) for lvl, lab in (("simple", 0.0), ("en", 1.0))]
    df = harmonize(coerce(pd.DataFrame(rows)))
    assert (df.loc[df.corpus == "wikipair", "mapping_confidence"] == 0).all()
    assert (df.loc[df.corpus == "clear", "mapping_confidence"] == 1).all()
    split = assign_splits(df, val_fraction=0.3, seed=0)
    assert (split.loc[split.corpus == "wikipair", "split"] == "train").all()  # never val


def test_group_shuffle_sampler_colocates_pair_members():
    from readability.training import _GroupShuffleSampler
    # groups: two 2-member articles + six singletons
    groups = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])
    sampler = _GroupShuffleSampler(groups, seed=0)
    order = list(iter(sampler))
    assert sorted(order) == list(range(10))             # a permutation
    for gid in (0, 1):                                  # pair members adjacent
        pos = [order.index(i) for i in np.flatnonzero(groups == gid)]
        assert abs(pos[0] - pos[1]) == 1


def test_rank_blend_ensemble_beats_both_singles():
    from analyze_transfer import rank_blend
    rng = np.random.default_rng(1)
    t = rng.normal(size=600)
    a = t + rng.normal(scale=1.0, size=600)   # two equally-noisy, independent views
    b = t + rng.normal(scale=1.0, size=600)
    h = rank_blend(a, b, w=0.5)
    assert spearman(t, h) > spearman(t, a)
    assert spearman(t, h) > spearman(t, b)    # classic ensemble gain


def test_calibration_probe_survives_stale_split_labels():
    from analyze_transfer import calibration_probe
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, 400)
    # split says 'train' (stale table, the 2026-07-03 LOCO symptom) -- the probe
    # must still run, grouped by corpus, and recover the constant -0.1 offset.
    df = pd.DataFrame({"corpus": "onestop", "split": "train",
                       "harmonized_difficulty": t, "pred": t - 0.1})
    out = calibration_probe(df, "harmonized_difficulty", k=50)
    assert len(out) == 1
    row = out.iloc[0]
    assert abs(row["signed_after"]) < abs(row["signed_before"])


def test_load_cefr_maps_levels(tmp_path):
    from readability.data import load_cefr
    csv = tmp_path / "cefr.csv"
    csv.write_text("text,label\nThe cat sat on the mat.,A1\n"
                   "Notwithstanding the epistemological ramifications elucidated.,C2\n", encoding="utf-8")
    out = load_cefr(csv)
    assert set(out["corpus"]) == {"cefr"}
    assert out["native_label"].min() == 0.0 and out["native_label"].max() == 5.0  # A1->0, C2->5


# --- Phase 4: external pool + pseudo-labeling (torch-free logic) ------------ #
def test_difficulty_proxy_orders_simple_below_complex():
    simple = "The cat sat. The dog ran. I see it."
    complex_ = "Notwithstanding the epistemological ramifications, the dissertation elucidated multifarious sociolinguistic considerations."
    assert difficulty_proxy(complex_) > difficulty_proxy(simple)


def test_select_diverse_caps_size_and_keeps_every_source():
    rows = [{"id": f"{src}:{i}", "text": ("word " * (5 + i % 40)).strip(), "corpus": src}
            for src in ("a", "b") for i in range(200)]
    out = select_diverse(pd.DataFrame(rows), n_total=60, n_bins=5, seed=0)
    assert len(out) <= 60
    assert set(out["corpus"]) == {"a", "b"}            # diversity: both sources survive


def test_build_external_pool_caps_chunks_per_doc(monkeypatch):
    import readability.external as ext
    long_doc = " ".join(["word"] * 2000)               # one long doc -> many raw chunks
    monkeypatch.setattr(ext, "fetch_texts", lambda source, limit: [long_doc])
    pool = ext.build_external_pool(["wiki_simple"], per_source_docs=1, max_chunks_per_doc=3)
    assert 1 <= len(pool) <= 3                          # capped to a few varied chunks per doc


def test_clear_bt_to_axis_inverts_easiness():
    gold = pd.DataFrame({"native_label": [-3.0, -1.0, 1.0],
                         "harmonized_difficulty": [1.0, 0.5, 0.0]})
    ax = clear_bt_to_axis(np.array([-3.0, 1.0]), gold)
    assert ax[0] > ax[1]                               # very negative BT (hard) -> high difficulty


def test_clear_bt_to_axis_extrapolates_not_clamps():
    gold = pd.DataFrame({"native_label": [-2.0, 0.0, 2.0], "harmonized_difficulty": [0.9, 0.5, 0.1]})
    ax = clear_bt_to_axis(np.array([-2.0, -3.0, -4.0]), gold, extrapolate=True)
    assert ax[1] > ax[0] and ax[2] > ax[1]             # harder-than-CLEAR stays ordered, not flattened
    assert ax[2] > 0.9                                 # extrapolated beyond the boundary
    clamped = clear_bt_to_axis(np.array([-3.0, -4.0]), gold, extrapolate=False)
    assert clamped[0] == clamped[1] == 0.9             # clamp flattens both onto the boundary


def test_generate_pseudo_labels_downweights_out_of_range():
    gold = coerce(pd.DataFrame({"id": ["clear:1", "clear:2"], "text": ["a", "b"], "corpus": "clear",
                                "native_label": [-1.0, 1.0], "harmonized_difficulty": [0.9, 0.1],
                                "std_error": [0.5, 0.5]}))
    pool = coerce(pd.DataFrame({"id": ["in:0", "out:0"], "text": ["p", "q"], "corpus": "x"}))
    gold_emb = np.array([[1.0, 0.0], [0.0, 1.0]]); pool_emb = np.array([[1.0, 0.3], [0.3, 1.0]])
    teacher_preds = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])  # in-range vs far out of CLEAR's range
    out = generate_pseudo_labels(pool, gold, teacher_preds=teacher_preds, pool_emb=pool_emb,
                                 gold_emb=gold_emb, k_se=100.0, max_std=10.0, dedup_cosine=0.0)
    cmap = dict(zip(out["id"], out["mapping_confidence"]))
    assert cmap["out:0"] < cmap["in:0"]               # teacher extrapolating -> lower confidence


def test_generate_pseudo_labels_se_filter_and_harmonize():
    gold = coerce(pd.DataFrame({
        "id": ["clear:1", "clear:2", "clear:3"], "text": ["a", "b", "c"], "corpus": "clear",
        "native_label": [-2.0, 0.0, 1.0], "harmonized_difficulty": [0.9, 0.5, 0.1],
        "std_error": [0.4, 0.4, 0.4]}))
    pool = coerce(pd.DataFrame({"id": ["x:0", "x:1"], "text": ["p", "q"], "corpus": "x"}))
    gold_emb = np.array([[1, 0], [0, 1], [1, 1]], float)
    pool_emb = np.array([[1, 0.4], [0.4, 1]], float)   # nearest clear:1 / clear:2, but not near-dups
    teacher_preds = np.array([[-2.0, -2.1, -1.9],      # x:0 plausible vs neighbour -> keep
                              [5.0, 5.1, 4.9]])         # x:1 implausible (|5-0|>se)  -> drop
    out = generate_pseudo_labels(pool, gold, teacher_preds=teacher_preds,
                                 pool_emb=pool_emb, gold_emb=gold_emb, k_se=1.0, max_std=1.0)
    assert set(out["id"]) == {"x:0"}
    assert bool(out["is_pseudo"].all())
    assert 0.0 <= float(out["harmonized_difficulty"].iloc[0]) <= 1.0


# --- Phase 8: ablation significance + aggregation --------------------------- #
def test_paired_bootstrap_detects_better_full():
    rng = np.random.default_rng(0)
    target = rng.normal(size=200)
    pred_full = target + rng.normal(scale=0.3, size=200)   # strongly correlated
    pred_variant = rng.normal(size=200)                    # ~uncorrelated
    s = paired_bootstrap_diff(target, pred_full, pred_variant, n_boot=500)
    assert s["delta"] > 0                                   # full ranks better
    assert s["p_full_not_better"] < 0.05                   # and it's significant


def test_aggregate_sorts_by_spearman_and_counts_seeds():
    rows = [{"variant": "full", "seed": 42, "spearman": 0.80, "rmse": 0.30},
            {"variant": "full", "seed": 43, "spearman": 0.82, "rmse": 0.29},
            {"variant": "no_pairwise", "seed": 42, "spearman": 0.70, "rmse": 0.35},
            {"variant": "no_pairwise", "seed": 43, "spearman": 0.72, "rmse": 0.34}]
    agg = aggregate(rows)
    assert agg["variant"].iloc[0] == "full"                # best mean Spearman first
    assert int(agg.loc[agg["variant"] == "full", "seeds"].iloc[0]) == 2


# --- Integrity fixes: torch seeding, cross-corpus dedup, near-dup gate -------- #
def test_seed_everything_makes_torch_reproducible():
    import torch
    seed_everything(123); a = torch.randn(8)
    seed_everything(123); b = torch.randn(8)
    assert torch.equal(a, b)                               # was non-reproducible before the fix


def test_dedup_against_drops_normalized_matches():
    pool = pd.DataFrame({"id": ["p1", "p2", "p3"],
                         "text": ["The cat sat.", "a unique passage", "  the   CAT  sat. "]})
    ref = pd.DataFrame({"text": ["the cat sat."]})
    out = dedup_against(pool, ref, key="text")
    assert set(out["id"]) == {"p2"}                        # p1 and p3 normalize to the reference


def test_generate_pseudo_labels_drops_near_duplicate_of_gold():
    gold = coerce(pd.DataFrame({"id": ["clear:1"], "text": ["a"], "corpus": "clear",
                                "native_label": [-2.0], "harmonized_difficulty": [0.9],
                                "std_error": [0.4]}))
    pool = coerce(pd.DataFrame({"id": ["dup:0"], "text": ["a"], "corpus": "x"}))
    gold_emb = np.array([[1.0, 0.0]]); pool_emb = np.array([[1.0, 0.0]])  # identical -> near-dup
    teacher_preds = np.array([[-2.0, -2.0, -2.0]])         # would pass SE, but it's a duplicate
    out = generate_pseudo_labels(pool, gold, teacher_preds=teacher_preds, pool_emb=pool_emb,
                                 gold_emb=gold_emb, k_se=1.0, max_std=1.0, dedup_cosine=0.05)
    assert len(out) == 0
