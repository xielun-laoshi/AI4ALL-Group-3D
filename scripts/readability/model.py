"""Model core (Phase 5): encoder backbone + regression head, with two upgrades
over the winner's single-task regressor:

  1. a per-source offset (label-aware / mixed-effects) so the model conditions on
     which corpus/scale a label came from and absorbs each source's systematic bias;
  2. a pairwise ranking signal (applied in the training loss), promoted to
     co-primary because rank transfers across corpora/formats far better than scale.

torch/transformers are required here (only imported when you do modeling); peft is
optional and imported lazily so a non-LoRA run needs no extra install.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# --------------------------------------------------------------------------- #
# Embedder: sentence vectors for diverse selection + SE-filter neighbour match #
# --------------------------------------------------------------------------- #
class Embedder:
    """Mean-pooled transformer embeddings. Used to (a) measure/sample for
    diversity in the external pool and (b) match each pseudo-labeled snippet to
    its nearest gold neighbour for the SE filter."""

    def __init__(self, backbone: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_length: int = 256, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModel.from_pretrained(backbone).to(self.device).eval()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs.append(pooled.float().cpu().numpy())
        return np.vstack(vecs)


# --------------------------------------------------------------------------- #
# DifficultyRegressor                                                          #
# --------------------------------------------------------------------------- #
class DifficultyRegressor(nn.Module):
    def __init__(self, backbone: str = "roberta-base", n_sources: int = 1,
                 dropout: float = 0.1, use_source_offset: bool = True,
                 n_extra_feats: int = 0) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Optional hand-crafted features (e.g. the standardized formula proxy)
        # concatenated to the pooled representation before the head. Motivated by
        # the per-corpus formula floors: the proxy alone scored 0.79 on CEFR and
        # 0.97 within-article on OneStop -- signal complementary to the encoder's.
        self.n_extra_feats = n_extra_feats
        self.head = nn.Linear(hidden + n_extra_feats, 1)
        self.use_source_offset = use_source_offset
        if use_source_offset:
            self.source_offset = nn.Embedding(max(n_sources, 1), 1)
            nn.init.zeros_(self.source_offset.weight)

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, source_id=None,
                extra_feats=None) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        if self.n_extra_feats:
            assert extra_feats is not None and extra_feats.shape[-1] == self.n_extra_feats, \
                "model built with n_extra_feats but none supplied"
            pooled = torch.cat([pooled, extra_feats.float()], dim=-1)
        score = self.head(self.dropout(pooled)).squeeze(-1)
        if self.use_source_offset and source_id is not None:
            score = score + self.source_offset(source_id).squeeze(-1)
        return score


def apply_lora(model: DifficultyRegressor, r: int = 16, alpha: int = 32,
               dropout: float = 0.05) -> DifficultyRegressor:
    """Wrap the encoder in LoRA adapters (optional; needs `peft`). The head and
    source offset stay fully trainable. On an H100 full fine-tuning is fine too --
    this is here for cheaper/faster runs and big backbones."""
    from peft import LoraConfig, get_peft_model

    targets = ["query_proj", "key_proj", "value_proj", "dense",  # deberta
               "query", "key", "value"]                           # bert/roberta
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=targets, bias="none")
    model.encoder = get_peft_model(model.encoder, cfg)
    return model


def build_tokenizer(backbone: str) -> "AutoTokenizer":
    return AutoTokenizer.from_pretrained(backbone)
