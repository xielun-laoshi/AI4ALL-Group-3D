"""Trainer (Phase 6 engine, also used to train the Phase-4 teacher).

Two-stage schedule lives in the calling scripts (pretrain on pseudo -> finetune on
gold); this module is the reusable fit/predict core. Loss = pointwise MSE
(sample-weighted by mapping_confidence) + optional pairwise ranking. bf16 autocast
on CUDA (H100-friendly); falls back to fp32 on CPU.

The teacher trains with target_col='native_label' on gold (predicting CLEAR BT so
the SE-filter works in BT units); the student trains with
target_col='harmonized_difficulty' on gold+pseudo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from .config import Config
from .evaluation import rmse, spearman
from .external import difficulty_proxy
from .model import DifficultyRegressor, apply_lora, build_tokenizer
from .utils import RunLogger, get_logger, seed_everything

log = get_logger("training")


class _Rows(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str, source_map: dict[str, int],
                 use_conf_weight: bool = True,
                 proxy_stats: tuple[float, float] | None = None):
        self.text = df["text"].astype(str).tolist()
        self.y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype="float32")
        self.src = df["corpus"].map(source_map).fillna(source_map.get("__unk__", 0)).to_numpy(dtype="int64")
        if use_conf_weight:
            w = pd.to_numeric(df.get("mapping_confidence", 1.0), errors="coerce").fillna(1.0)
            self.w = np.clip(w.to_numpy(dtype="float32"), 0.0, 1.0)
        else:
            self.w = np.ones(len(self.text), dtype="float32")
        if proxy_stats is not None:   # formula feature, standardized by TRAIN stats
            mean, std = proxy_stats
            raw = np.array([difficulty_proxy(t) for t in self.text], dtype="float32")
            self.feat = (raw - mean) / max(std, 1e-6)
        else:
            self.feat = np.zeros(len(self.text), dtype="float32")
        self.ids = df["id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i: int):
        return self.text[i], self.y[i], self.src[i], self.w[i], self.feat[i], self.ids[i]


def _pairwise_loss(scores: torch.Tensor, targets: torch.Tensor,
                   sources: torch.Tensor | None = None, margin: float = 0.0) -> torch.Tensor:
    """Margin ranking loss over ordered in-batch pairs where the targets differ --
    teaches correct *ordering*, which transfers across corpora.

    Pairs are restricted to the SAME source when ``sources`` is given: percentile
    targets from different corpora aren't on a comparable ruler (0.5 in CEFR is
    not 0.5 in CLEAR), so cross-source pairs inject scale-mismatch noise -- the
    run-2 ablation measured the unmasked head as net-negative for transfer."""
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())
    si, sj = scores.unsqueeze(0), scores.unsqueeze(1)
    ti, tj = targets.unsqueeze(0), targets.unsqueeze(1)
    sign = torch.sign(ti - tj)
    mask = sign != 0
    if sources is not None:
        mask = mask & (sources.unsqueeze(0) == sources.unsqueeze(1))
    if mask.sum() == 0:
        return scores.new_zeros(())
    diff = (si - sj)
    loss = torch.clamp(margin - sign * diff, min=0.0)
    return loss[mask].mean()


class Trainer:
    def __init__(self, cfg: Config, *, target_col: str = "harmonized_difficulty",
                 backbone: str | None = None) -> None:
        self.cfg = cfg
        self.target_col = target_col
        self.backbone = backbone or cfg.model.backbone
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = build_tokenizer(self.backbone)
        self.source_map: dict[str, int] = {}
        self.proxy_stats: tuple[float, float] | None = None  # set when formula feature on
        self.model: DifficultyRegressor | None = None
        seed_everything(cfg.train.seed)

    # --- internals --------------------------------------------------------- #
    def _collate(self, batch):
        texts, ys, srcs, ws, feats, ids = zip(*batch)
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.cfg.model.max_length, return_tensors="pt")
        return (enc, torch.tensor(ys), torch.tensor(srcs), torch.tensor(ws),
                torch.tensor(feats), list(ids))

    def _loader(self, df, *, shuffle: bool, batch_size: int) -> DataLoader:
        ds = _Rows(df, self.target_col, self.source_map,
                   use_conf_weight=self.cfg.train.confidence_weighting,
                   proxy_stats=self.proxy_stats)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._collate, num_workers=0, pin_memory=(self.device == "cuda"))

    def _feats_arg(self, feats: torch.Tensor):
        """[B] -> [B,1] on device when the formula feature is enabled, else None."""
        if self.proxy_stats is None:
            return None
        return feats.unsqueeze(-1).to(self.device)

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)

    def _transfer_from(self, ckpt: str | Path) -> None:
        """Warm-start: copy matching-shape tensors from a checkpoint. The encoder
        and head transfer across stages; the per-source offset (sized to a
        different source_map) is left fresh."""
        blob = torch.load(ckpt, map_location=self.device)
        own = self.model.state_dict()
        transfer = {k: v for k, v in blob["state_dict"].items()
                    if k in own and own[k].shape == v.shape}
        own.update(transfer)
        self.model.load_state_dict(own)
        log.info("init_from: transferred %d/%d tensors (encoder+head kept, source offset reset)",
                 len(transfer), len(own))

    # --- public API -------------------------------------------------------- #
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None,
            run: RunLogger | None = None, init_from: str | Path | None = None) -> "Trainer":
        tc = self.cfg.train
        corpora = sorted(train_df["corpus"].unique())
        # Reserve an "__unk__" source whose offset stays ~0 (no training row maps to
        # it, so it gets no gradient). Unseen corpora at predict time then get the
        # population-mean offset instead of some seen corpus's bias -- keeps
        # cross-corpus absolute error honest.
        self.source_map = {c: i for i, c in enumerate(corpora)}
        self.source_map["__unk__"] = len(corpora)
        if self.cfg.model.use_formula_feature:
            raw = np.array([difficulty_proxy(t) for t in train_df["text"].astype(str)])
            self.proxy_stats = (float(raw.mean()), float(raw.std()))
            log.info("formula feature on: proxy mean=%.2f std=%.2f (train)", *self.proxy_stats)
        self.model = DifficultyRegressor(self.backbone, n_sources=len(self.source_map),
                                         dropout=self.cfg.model.dropout,
                                         use_source_offset=self.cfg.model.use_source_offset,
                                         n_extra_feats=1 if self.proxy_stats is not None else 0)
        if self.cfg.model.peft:
            self.model = apply_lora(self.model)
        self.model.to(self.device)
        if init_from is not None:                # two-stage: warm-start from Stage A
            self._transfer_from(init_from)

        opt = torch.optim.AdamW(self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        loader = self._loader(train_df, shuffle=True, batch_size=tc.batch_size)
        total_steps = max(tc.epochs * len(loader), 1)
        sched = get_cosine_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
        log.info("fit: backbone=%s rows=%d sources=%d device=%s target=%s steps=%d",
                 self.backbone, len(train_df), len(corpora), self.device, self.target_col, total_steps)

        best_rmse, best_state = float("inf"), None
        for epoch in range(tc.epochs):
            self.model.train()
            running = 0.0
            for enc, y, src, w, feats, _ in loader:
                enc = {k: v.to(self.device) for k, v in enc.items()}
                y, src, w = y.to(self.device), src.to(self.device), w.to(self.device)
                opt.zero_grad()
                with self._autocast():
                    pred = self.model(enc["input_ids"], enc["attention_mask"], src,
                                      extra_feats=self._feats_arg(feats))
                    point = (w * (pred - y) ** 2).mean() * tc.pointwise_weight
                    pair = _pairwise_loss(pred, y, src) * tc.pairwise_weight if self.cfg.model.use_pairwise_head else 0.0
                    loss = point + pair
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                running += float(loss) * len(y)
            msg = {"epoch": epoch + 1, "train_loss": running / len(train_df)}
            if val_df is not None and len(val_df):
                vp = self.predict(val_df)["pred"].to_numpy()
                vy = pd.to_numeric(val_df[self.target_col], errors="coerce").to_numpy()
                msg["val_rmse"], msg["val_spearman"] = rmse(vy, vp), spearman(vy, vp)
                if msg["val_rmse"] < best_rmse:          # keep the best epoch, not the last
                    best_rmse = msg["val_rmse"]
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            log.info("epoch %d | %s", epoch + 1, {k: round(v, 4) for k, v in msg.items() if k != "epoch"})
            if run:
                run.log_metrics(msg, step=epoch + 1)
        if best_state is not None:
            self.model.load_state_dict(best_state)
            log.info("restored best epoch (val_rmse=%.4f)", best_rmse)
        return self

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, batch_size: int | None = None) -> pd.DataFrame:
        assert self.model is not None, "call fit() or load() first"
        self.model.eval()
        loader = self._loader(df, shuffle=False, batch_size=batch_size or self.cfg.train.batch_size * 2)
        preds, ids = [], []
        for enc, _y, src, _w, feats, batch_ids in loader:
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self._autocast():
                out = self.model(enc["input_ids"], enc["attention_mask"], src.to(self.device),
                                 extra_feats=self._feats_arg(feats))
            preds.append(out.float().cpu().numpy())
            ids.extend(batch_ids)
        return pd.DataFrame({"id": ids, "pred": np.concatenate(preds)})

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "source_map": self.source_map,
                    "backbone": self.backbone, "target_col": self.target_col,
                    "proxy_stats": self.proxy_stats}, out / "model.pt")
        log.info("saved -> %s", out / "model.pt")
        return out / "model.pt"

    def load(self, ckpt: str | Path) -> "Trainer":
        blob = torch.load(ckpt, map_location=self.device)
        self.source_map = blob["source_map"]
        self.backbone = blob["backbone"]
        self.target_col = blob["target_col"]
        self.proxy_stats = blob.get("proxy_stats")   # honor the checkpoint, not cfg
        self.model = DifficultyRegressor(self.backbone, n_sources=len(self.source_map),
                                         use_source_offset=self.cfg.model.use_source_offset,
                                         n_extra_feats=1 if self.proxy_stats is not None else 0)
        if self.cfg.model.peft:
            self.model = apply_lora(self.model)
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device)
        return self


def fit_student(cfg: Config, pool_df: pd.DataFrame, *,
                target_col: str = "harmonized_difficulty",
                run: RunLogger | None = None) -> Trainer:
    """Train the student: single-pass mixed (default) or the strict two-stage
    schedule -- pretrain on the large pseudo pool at low LR, then finetune on gold,
    warm-starting from Stage A. Returns the fitted Trainer.

    Used by both scripts/train.py and the Phase-8 harness, so 'two-stage vs
    single-pass' is itself a togglable ablation (train.two_stage)."""
    import copy
    import tempfile

    if not cfg.train.two_stage:
        return Trainer(cfg, target_col=target_col).fit(
            pool_df[pool_df["split"] == "train"], pool_df[pool_df["split"] == "val"], run=run)

    pseudo = pool_df[pool_df["is_pseudo"] == True]
    gold = pool_df[pool_df["is_pseudo"] != True]
    if len(pseudo) == 0:
        log.warning("two_stage requested but the pool has no pseudo rows; single-pass instead")
        return Trainer(cfg, target_col=target_col).fit(
            gold[gold["split"] == "train"], gold[gold["split"] == "val"], run=run)

    # Stage A: pretrain on the large pseudo pool at a low LR (absorb broad difficulty)
    cfg_a = copy.deepcopy(cfg)
    cfg_a.train.lr, cfg_a.train.epochs = cfg.train.pretrain_lr, cfg.train.pretrain_epochs
    log.info("two-stage A: pretrain on %d pseudo rows (lr=%.1e epochs=%d)",
             len(pseudo), cfg_a.train.lr, cfg_a.train.epochs)
    stage_a = Trainer(cfg_a, target_col=target_col).fit(pseudo, None, run=run)

    # Stage B: finetune on gold, warm-started from Stage A's encoder (anchor to truth)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = stage_a.save(tmp)
        log.info("two-stage B: finetune on %d gold rows (lr=%.1e epochs=%d)",
                 len(gold), cfg.train.lr, cfg.train.epochs)
        return Trainer(cfg, target_col=target_col).fit(
            gold[gold["split"] == "train"], gold[gold["split"] == "val"], run=run, init_from=ckpt)
