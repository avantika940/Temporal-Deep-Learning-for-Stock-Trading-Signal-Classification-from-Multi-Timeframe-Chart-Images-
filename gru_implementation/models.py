"""
BiGRU model variants for stock signal classification.

Five architectures that mirror the BiLSTM suite but replace every LSTM cell
with a GRU cell.  GRU advantages over LSTM:
  • ~33 % fewer parameters (no separate cell state / output gate)
  • Faster training per epoch
  • Comparable or better performance on shorter sequences

Models:
  1. DeepBiGRU       — 3-layer stacked BiGRU
  2. AttentionBiGRU  — BiGRU + self-attention pooling
  3. ResidualBiGRU   — BiGRU with residual + layer-norm skip connections
  4. HybridBiGRU     — Residual BiGRU + multi-head attention + gated fusion  ⭐ BEST
  5. PyramidalBiGRU  — Hierarchical BiGRU + temporal pyramid pooling

Baseline reference: Step-4 Temporal Transformer → 61.21 % val accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. DeepBiGRU
# ─────────────────────────────────────────────────────────────────────────────

class DeepBiGRU(nn.Module):
    """
    Deep stacked BiGRU baseline.

    Architecture:
      Frozen ViT  →  GRU-1(256)  →  GRU-2(256)  →  GRU-3(128)
      → last hidden state  →  FC classifier

    Key difference from LSTM: GRU output is (output, h_n) — no cell state.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.gru1 = nn.GRU(
            input_size=config.VIT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru2 = nn.GRU(
            input_size=config.HIDDEN_DIM * 2,
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru3 = nn.GRU(
            input_size=config.HIDDEN_DIM * 2,
            hidden_size=config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(config.GRU_DROPOUT)

        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES),
        )

    def forward(self, x):
        """x: [batch, seq_len, C, H, W]"""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            features = self.vit(x).last_hidden_state[:, 0, :]   # CLS token
        features = features.view(B, T, -1)

        x, _ = self.gru1(features);  x = self.dropout(x)
        x, _ = self.gru2(x);         x = self.dropout(x)
        x, _ = self.gru3(x)

        return self.classifier(x[:, -1, :])   # last time-step


# ─────────────────────────────────────────────────────────────────────────────
# 2. AttentionBiGRU
# ─────────────────────────────────────────────────────────────────────────────

class AttentionBiGRU(nn.Module):
    """
    BiGRU with additive self-attention pooling.

    Architecture:
      Frozen ViT  →  2-layer BiGRU  →  self-attention weighted sum  →  classifier
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.bigru = nn.GRU(
            input_size=config.VIT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            dropout=config.GRU_DROPOUT,
            bidirectional=True,
        )

        self.attention = SelfAttention(config.HIDDEN_DIM * 2, config.ATTENTION_DROPOUT)

        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            features = self.vit(x).last_hidden_state[:, 0, :]
        features = features.view(B, T, -1)

        gru_out, _ = self.bigru(features)
        context, _ = self.attention(gru_out)
        return self.classifier(context)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ResidualBiGRU
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBiGRU(nn.Module):
    """
    BiGRU with residual (skip) connections and layer normalisation.

    Architecture:
      Frozen ViT  →  linear projection  →  [GRU + residual + LN] × 3
      → last hidden state  →  classifier
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for p in self.vit.parameters():
                p.requires_grad = False

        # Project ViT dim  →  HIDDEN_DIM * 2 so residual shapes match
        self.input_proj = nn.Linear(config.VIT_DIM, config.HIDDEN_DIM * 2)

        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=config.HIDDEN_DIM * 2,
                hidden_size=config.HIDDEN_DIM,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            for _ in range(3)
        ])

        if config.USE_LAYER_NORM:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(config.HIDDEN_DIM * 2) for _ in range(3)
            ])
        else:
            self.layer_norms = None

        self.dropout = nn.Dropout(config.RESIDUAL_DROPOUT)

        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            features = self.vit(x).last_hidden_state[:, 0, :]
        features = features.view(B, T, -1)

        x = self.input_proj(features)

        for i, gru in enumerate(self.gru_layers):
            residual = x
            x, _ = gru(x)
            x = self.dropout(x)
            x = x + residual
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)

        return self.classifier(x[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# 4. HybridBiGRU  ⭐  (expected best)
# ─────────────────────────────────────────────────────────────────────────────

class HybridBiGRU(nn.Module):
    """
    Hybrid model: residual BiGRU + multi-head attention + gated fusion.

    Architecture:
      Frozen ViT  →  linear projection
      →  [GRU + residual + LN] × 3              (temporal encoding)
      →  multi-head self-attention               (global context)
      →  gated fusion (learn mixture of both)
      →  mean-pool over sequence  →  classifier
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.input_proj = nn.Linear(config.VIT_DIM, config.HIDDEN_DIM * 2)

        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=config.HIDDEN_DIM * 2,
                hidden_size=config.HIDDEN_DIM,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            for _ in range(3)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.HIDDEN_DIM * 2) for _ in range(3)
        ])

        self.dropout = nn.Dropout(config.GRU_DROPOUT)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM * 2,
            num_heads=config.ATTENTION_HEADS,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True,
        )

        # Gated fusion: decides how much of GRU vs attention to use
        self.gate = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 4, config.HIDDEN_DIM * 2),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            features = self.vit(x).last_hidden_state[:, 0, :]
        features = features.view(B, T, -1)

        x = self.input_proj(features)

        # Residual BiGRU stack
        for i, gru in enumerate(self.gru_layers):
            residual = x
            x, _ = gru(x)
            x = self.dropout(x)
            x = x + residual
            x = self.layer_norms[i](x)

        gru_out = x

        # Multi-head self-attention
        attn_out, _ = self.multihead_attn(x, x, x)

        # Gated fusion
        gate_v = self.gate(torch.cat([gru_out, attn_out], dim=-1))
        fused  = gate_v * gru_out + (1 - gate_v) * attn_out

        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PyramidalBiGRU
# ─────────────────────────────────────────────────────────────────────────────

class PyramidalBiGRU(nn.Module):
    """
    Pyramidal BiGRU with decreasing hidden sizes and temporal pyramid pooling.

    Architecture:
      Frozen ViT  →  GRU-1(512)  →  GRU-2(256)  →  GRU-3(128)
      →  temporal pyramid pool [1, 2, 5]  →  classifier

    Multi-scale pooling captures both local (fine) and global (coarse) patterns.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.gru1 = nn.GRU(
            input_size=config.VIT_DIM,
            hidden_size=config.HIDDEN_DIM * 2,    # 512
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru2 = nn.GRU(
            input_size=config.HIDDEN_DIM * 4,     # 512*2 bidirectional
            hidden_size=config.HIDDEN_DIM,         # 256
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru3 = nn.GRU(
            input_size=config.HIDDEN_DIM * 2,     # 256*2
            hidden_size=config.HIDDEN_DIM // 2,   # 128
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(config.GRU_DROPOUT)

        # Pyramid pool: pool sizes [1, 2, 5] → sum = 8 banks × HIDDEN_DIM features
        self.pyramid = TemporalPyramidPooling(config.HIDDEN_DIM, pool_sizes=[1, 2, 5])
        pooled_dim = config.HIDDEN_DIM * (1 + 2 + 5)   # 256 * 8 = 2048

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            features = self.vit(x).last_hidden_state[:, 0, :]
        features = features.view(B, T, -1)

        x, _ = self.gru1(features);  x = self.dropout(x)
        x, _ = self.gru2(x);         x = self.dropout(x)
        x, _ = self.gru3(x)

        x = self.pyramid(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Helper modules
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Additive (Bahdanau-style) self-attention over a sequence."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, T, H]  →  context: [B, H], weights: [B, T]"""
        scores  = self.attn(x).squeeze(-1)                      # [B, T]
        weights = F.softmax(scores, dim=1)                       # [B, T]
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1) # [B, H]
        return self.dropout(context), weights


class TemporalPyramidPooling(nn.Module):
    """Multi-scale temporal pooling producing a fixed-length descriptor."""

    def __init__(self, hidden_dim: int, pool_sizes=(1, 2, 5)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """x: [B, T, H]  →  [B, H * sum(pool_sizes)]"""
        B, T, H = x.shape
        parts = []
        for ps in self.pool_sizes:
            if ps == 1:
                parts.append(x.mean(dim=1))              # [B, H]
            else:
                chunk = T // ps
                chunks = [
                    x[:, i * chunk: (i + 1) * chunk if i < ps - 1 else T, :].mean(dim=1)
                    for i in range(ps)
                ]
                parts.append(torch.cat(chunks, dim=-1))  # [B, H * ps]
        return torch.cat(parts, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    'deep':      DeepBiGRU,
    'attention': AttentionBiGRU,
    'residual':  ResidualBiGRU,
    'hybrid':    HybridBiGRU,
    'pyramidal': PyramidalBiGRU,
}


def create_model(config):
    """
    Instantiate the requested BiGRU model.

    Args:
        config: Config object with MODEL_TYPE and hyperparameters.

    Returns:
        nn.Module instance.
    """
    cls = _MODEL_REGISTRY.get(config.MODEL_TYPE.lower())
    if cls is None:
        raise ValueError(
            f"Unknown model type '{config.MODEL_TYPE}'. "
            f"Choose from: {list(_MODEL_REGISTRY.keys())}"
        )

    model = cls(config)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: {config.MODEL_TYPE.upper()} BiGRU")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters:    {total - trainable:,}")

    return model
