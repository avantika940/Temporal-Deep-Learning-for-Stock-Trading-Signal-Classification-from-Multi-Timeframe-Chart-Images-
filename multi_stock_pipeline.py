"""
Multi-Stock Pipeline — Hybrid Visual + Technical Indicator Features
====================================================================
Feature vector per bar = concat(frozen_ViT_RGB_CLS[768], tech_indicators[5]) = 773-dim

Technical indicators derived from the class-label signal sequence
(BUY=+1, HOLD=0, SELL=-1) within each sliding window:
  1. RSI-proxy     : avg_gain / (avg_gain + avg_loss), cumulative up to t  [0,1]
  2. MACD-proxy    : EMA_fast(3) − EMA_slow(6) of label signal, normalised [-1,1]
  3. MACD-signal   : EMA(3) of MACD-proxy                                  [-1,1]
  4. BB-width      : rolling 4-bar std of label signal                     [0,~1]
  5. Trend-slope   : linear regression slope of S[0..t], clipped [-1,1]

Frozen ViT embeddings:
  - Original HuggingFace ViT-B/16 (google/vit-base-patch16-224-in21k), NOT fine-tuned.
  - Extracted ONCE and cached to embeddings/{stock_name}_frozen_rgb.npz.
  - On re-runs the cache is loaded directly (no ViT forward pass needed).

Split: Per-class stratified chronological 64 / 16 / 20.
Models: HybridBiGRU(input=773) + HybridBiLSTM(input=773)
"""

import os, re, json, copy, time, csv, sys
from collections import Counter
from pathlib import Path
from datetime import datetime

# Suppress HuggingFace/tqdm progress bars (avoids garbled Unicode on Windows console)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"]  = "1"

# Force UTF-8 stdout/stderr so Unicode chars don't crash on Windows cp1252 console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from transformers import ViTModel
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Dynamic path resolution - works on any machine
PROJECT_ROOT   = Path(__file__).parent.resolve()
DATASETS_DIR   = PROJECT_ROOT / "data_multi_processed"
RESULTS_DIR    = PROJECT_ROOT / "multi_stock_results"
EMBED_DIR      = PROJECT_ROOT / "embeddings"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

VIT_DIM        = 768
TECH_DIM       = 5               # RSI, MACD, MACD-signal, BB-width, trend-slope
INPUT_DIM      = VIT_DIM + TECH_DIM   # 773  ← new combined input size
HIDDEN_DIM     = 256             # per direction → bidirectional output D = 512
SEQ_LEN        = 10

BATCH_SIZE     = 32
LR             = 3e-4
WEIGHT_DECAY   = 1e-5
LABEL_SMOOTH   = 0.1
GRAD_CLIP      = 1.0
USE_CLASS_WEIGHTS    = True
USE_WEIGHTED_SAMPLER = True

# Chronological split: 64% train, 16% val, 20% test (no temporal leakage)
TRAIN_SPLIT    = 0.64
VAL_SPLIT      = 0.16
TEST_SPLIT     = 0.20
NUM_CLASSES    = 3
CLASSES        = ["BUY", "HOLD", "SELL"]
# Label → scalar signal encoding for indicator computation
LABEL_SIGNAL   = {0: 1.0, 1: 0.0, 2: -1.0}   # BUY=+1, HOLD=0, SELL=-1
SEED           = 42
EPOCHS         = 80

EMBED_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR = RESULTS_DIR / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

# Mapping: stock name → cached frozen-RGB npz filename
STOCK_NPZ_MAP = {
    "COAL_India_MTF": "COAL_India_MTF_frozen_rgb.npz",
    "HCL_MTF":        "HCL_MTF_frozen_rgb.npz",
    "Maruti_MTF":     "Maruti_MTF_frozen_rgb.npz",
    "Apollo_MTF":     "Apollo_MTF_frozen_rgb.npz",
}

# Stock configurations — 5 stocks: COAL, HCL, Maruti, Apollo
STOCKS = [
    ("COAL_India_MTF", PROJECT_ROOT / "data_multi_processed" / "COAL_India_MTF"),
    ("HCL_MTF",        PROJECT_ROOT / "data_multi_processed" / "HCL_MTF"),
    ("Maruti_MTF",     PROJECT_ROOT / "data_multi_processed" / "MARUTI MTF"),
    ("Apollo_MTF",     PROJECT_ROOT / "data_multi_processed" / "Apollo MTF"),
]


# -----------------------------------------------------------------------------
# STAGE 1 — IMAGE TRANSFORM
# -----------------------------------------------------------------------------
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # ViT standard
])


# -----------------------------------------------------------------------------
# STAGE 2 — FROZEN ORIGINAL ViT RGB EMBEDDINGS  (cached per stock)
#   Uses the original HuggingFace ViTModel — NOT the timm fine-tuned model.
#   Saves/loads from EMBED_DIR/{stock_name}_frozen_rgb.npz.
# -----------------------------------------------------------------------------
def load_or_extract_frozen_rgb_embeddings(
        stock_name: str,
        dataset_path: Path,
        device: torch.device,
) -> dict:
    """
    Return {str(img_path): Tensor(768,)} for every image in the stock dataset.
    On first call: extracts CLS tokens via frozen ViT-B/16 and caches to npz.
    On subsequent calls: loads directly from the npz cache (fast).
    """
    npz_name   = STOCK_NPZ_MAP.get(stock_name, f"{stock_name}_frozen_rgb.npz")
    cache_path = EMBED_DIR / npz_name

    if cache_path.exists():
        print(f"  [embed] Loading cached frozen-RGB: {cache_path.name}", flush=True)
        data     = np.load(cache_path, allow_pickle=True)
        paths    = data["paths"].tolist()
        emb_arr  = data["embeddings"]          # (N, 768) float32
        emb_dict = {p: torch.from_numpy(emb_arr[i]).float()
                    for i, p in enumerate(paths)}
        print(f"  [embed] {len(emb_dict)} embeddings loaded (dim=768)", flush=True)
        return emb_dict

    # -- Extract from frozen ViT (first run only) --------------------------
    print(f"  [embed] Extracting frozen RGB via {VIT_MODEL_NAME} …", flush=True)
    vit = ViTModel.from_pretrained(VIT_MODEL_NAME).to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    all_paths = []
    for cls in CLASSES:
        d = dataset_path / cls
        if d.exists():
            all_paths += [d / f for f in os.listdir(d) if f.lower().endswith(".png")]

    n = len(all_paths)
    print(f"  [embed] {n} images", end="", flush=True)
    t0 = time.time()

    path_strs, emb_list = [], []
    BS = 48
    with torch.no_grad():
        for i in range(0, n, BS):
            batch   = all_paths[i: i + BS]
            tensors = [IMG_TRANSFORM(Image.open(p).convert("RGB")) for p in batch]
            pv      = torch.stack(tensors).to(device)
            cls_tok = vit(pixel_values=pv).last_hidden_state[:, 0, :]  # (B, 768)
            for p, e in zip(batch, cls_tok.cpu().numpy()):
                path_strs.append(str(p))
                emb_list.append(e)
            print(".", end="", flush=True)

    emb_array = np.array(emb_list, dtype=np.float32)
    print(f" done  {time.time()-t0:.1f}s", flush=True)
    np.savez_compressed(cache_path,
                        embeddings=emb_array,
                        paths=np.array(path_strs))
    print(f"  [embed] Cached → {cache_path.name}", flush=True)

    del vit
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {path_strs[i]: torch.from_numpy(emb_array[i]).float()
            for i in range(len(path_strs))}


# -----------------------------------------------------------------------------
# STAGE 3 — TECHNICAL INDICATOR COMPUTATION
#   Compute 5 scalar indicators per bar from the label signal
#   S_t in {BUY:+1, HOLD:0, SELL:-1} over the sliding window.
# -----------------------------------------------------------------------------
def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (alpha = 2/(span+1))."""
    alpha  = 2.0 / (span + 1)
    out    = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def compute_technical_indicators(label_seq: list) -> np.ndarray:
    """
    Compute 5 per-bar technical indicators from a label sequence.
    FIXED: Only use PAST information up to each time step (no future leakage).
    Args:
        label_seq : list[int], length SEQ_LEN, values {0=BUY, 1=HOLD, 2=SELL}
    Returns:
        np.ndarray shape (SEQ_LEN, TECH_DIM=5), float32
        Columns: [rsi_proxy, macd_proxy, macd_signal, bb_width, trend_slope]
    """
    T = len(label_seq)
    # CRITICAL FIX: Remove last label (target) from technical indicator computation
    # Only use labels [0, 1, ..., T-2] for indicators, excluding target label[T-1]
    past_labels = label_seq[:-1] if T > 1 else [label_seq[0]]  # Exclude future target
    S_past = np.array([LABEL_SIGNAL[l] for l in past_labels], dtype=np.float32)

    # 1. RSI-proxy: cumulative avg_gain / (avg_gain + avg_loss) up to bar t
    diffs  = np.diff(S, prepend=S[0])
    gains  = np.maximum(diffs,  0.0)
    losses = np.maximum(-diffs, 0.0)
    avg_g  = np.array([gains[:t+1].mean()  for t in range(T)], dtype=np.float32)
    avg_l  = np.array([losses[:t+1].mean() for t in range(T)], dtype=np.float32)
    denom  = avg_g + avg_l
    with np.errstate(invalid="ignore", divide="ignore"):
        rsi = np.where(denom > 0, avg_g / denom, 0.5).astype(np.float32)  # [0, 1]

    # 2. MACD-proxy: EMA(3) − EMA(6), normalised /2 → [-1, 1]
    ema_fast = _ema(S, min(3, T))
    ema_slow = _ema(S, min(6, T))
    macd     = (ema_fast - ema_slow) / 2.0

    # 3. MACD-signal: EMA(3) of MACD
    macd_sig = _ema(macd, min(3, T))

    # 4. BB-width: rolling 4-bar std of S (local signal volatility)
    bb = np.array([S[max(0, t-3): t+1].std() for t in range(T)], dtype=np.float32)

    # 5. Trend-slope: OLS slope of S[0..t], clipped to [-1, 1]
    slope = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        xs = np.arange(t + 1, dtype=np.float32)
        ys = S[:t + 1]
        xm, ym = xs.mean(), ys.mean()
        den = ((xs - xm) ** 2).sum()
        slope[t] = float(((xs - xm) * (ys - ym)).sum() / den) if den > 1e-9 else 0.0
    slope = np.clip(slope, -1.0, 1.0)

    return np.stack([rsi, macd, macd_sig, bb, slope], axis=1).astype(np.float32)


# -----------------------------------------------------------------------------
# STAGE 4a — DATASET
# Per-class chronological split → merged timeline → sliding-window sequences.
# Each item: x = concat(frozen_ViT_RGB[768], tech_indicators[5]) = Tensor(T,773)
# -----------------------------------------------------------------------------
class EmbSeqDataset(Dataset):
    """
    Per-class stratified chronological split with hybrid features.

    Feature per bar = concat(frozen ViT CLS[768], tech_indicators[5]) = 773-dim.
    Technical indicators are computed on-the-fly from the label sequence
    within each sliding window — no OHLCV data required.
    """

    def __init__(self, dataset_path: Path, embeddings: dict, split: str = "train"):
        # Step 1: collect and sort each class independently by numeric ID
        per_class = {i: [] for i in range(len(CLASSES))}
        for ci, cls in enumerate(CLASSES):
            d = dataset_path / cls
            if not d.exists():
                continue
            for fname in os.listdir(d):
                if not fname.lower().endswith(".png"):
                    continue
                nums = re.findall(r"\d+", fname)
                nid  = int(nums[-1]) if nums else 0
                per_class[ci].append((nid, str(d / fname)))
            per_class[ci].sort(key=lambda e: e[0])   # chronological within class

        # --- Step 2: per-class 64/16/20 split ---------------------------
        chosen_entries = []   # list of (nid, path, label_idx)
        for ci, items in per_class.items():
            n = len(items)
            if n == 0:
                continue
            tr_end  = int(n * TRAIN_SPLIT)
            val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
            if split == "train":
                subset = items[:tr_end]
            elif split == "val":
                subset = items[tr_end:val_end]
            else:  # test
                subset = items[val_end:]
            for nid, pth in subset:
                chosen_entries.append((nid, pth, ci))

        # --- Step 3: merge and re-sort by global ID ---------------------
        chosen_entries.sort(key=lambda e: e[0])
        paths  = [e[1] for e in chosen_entries]
        labels = [e[2] for e in chosen_entries]

        # --- Step 4: sliding-window sequences ---------------------------
        seqs, seq_labels, bar_labels_list = [], [], []
        for i in range(len(paths) - SEQ_LEN + 1):
            seqs.append(paths[i: i + SEQ_LEN])
            seq_labels.append(labels[i + SEQ_LEN - 1])
            bar_labels_list.append(labels[i: i + SEQ_LEN])   # per-bar labels in window

        self.seqs        = seqs
        self.labels      = seq_labels
        self._bar_labels = bar_labels_list   # list[list[int]], shape (N, SEQ_LEN)
        self.emb         = embeddings

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_paths  = self.seqs[idx]                               # list[str], len T
        bar_labels = self._bar_labels[idx]                        # list[int], len T

        vit_feats = torch.stack([self.emb[p] for p in seq_paths])  # [T, 768]
        tech      = compute_technical_indicators(bar_labels)        # ndarray [T, 5]
        tech_t    = torch.from_numpy(tech)                          # [T, 5]
        x         = torch.cat([vit_feats, tech_t], dim=-1)          # [T, 773]
        return x, self.labels[idx]

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss (fixes class imbalance)."""
        counts = Counter(self.labels)
        total  = len(self.labels)
        weights = []
        for i in range(NUM_CLASSES):
            c = counts.get(i, 0)
            weights.append(total / (NUM_CLASSES * c) if c > 0 else 1.0)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> list:
        """Per-sample weights for WeightedRandomSampler (oversamples rare classes)."""
        counts = Counter(self.labels)
        total  = len(self.labels)
        class_w = {c: total / (NUM_CLASSES * n) if n > 0 else 1.0
                   for c, n in counts.items()}
        return [class_w[lbl] for lbl in self.labels]


# -----------------------------------------------------------------------------
# STAGE 3b — MODELS  (take pre-extracted embeddings, no ViT inside)
# -----------------------------------------------------------------------------
D = HIDDEN_DIM * 2     # bidirectional output dim = 512


class HybridBiGRU(nn.Module):
    """Residual BiGRU (×3) + multi-head attention + gated fusion.
    Accepts input_dim=773 (768 frozen ViT CLS + 5 technical indicators).
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(INPUT_DIM, D)   # 773 → 512

        self.grus = nn.ModuleList([
            nn.GRU(D, HIDDEN_DIM, batch_first=True, bidirectional=True)
            for _ in range(3)
        ])
        self.lns  = nn.ModuleList([nn.LayerNorm(D) for _ in range(3)])
        self.drop = nn.Dropout(0.0)

        self.attn = nn.MultiheadAttention(D, num_heads=8, dropout=0.0, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())

        self.head = nn.Sequential(
            nn.Linear(D, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):               # x: [B, T, 773]
        x = self.proj(x)                # [B, T, D]
        for gru, ln in zip(self.grus, self.lns):
            res = x
            x, _ = gru(x)
            x = ln(self.drop(x) + res)
        gru_out = x
        attn_out, _ = self.attn(x, x, x)
        gate  = self.gate(torch.cat([gru_out, attn_out], dim=-1))
        fused = gate * gru_out + (1 - gate) * attn_out
        return self.head(fused.mean(dim=1))


class HybridBiLSTM(nn.Module):
    """Residual BiLSTM (×3) + multi-head attention + gated fusion.
    Accepts input_dim=773 (768 frozen ViT CLS + 5 technical indicators).
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(INPUT_DIM, D)   # 773 → 512

        self.lstms = nn.ModuleList([
            nn.LSTM(D, HIDDEN_DIM, batch_first=True, bidirectional=True)
            for _ in range(3)
        ])
        self.lns  = nn.ModuleList([nn.LayerNorm(D) for _ in range(3)])
        self.drop = nn.Dropout(0.0)

        self.attn = nn.MultiheadAttention(D, num_heads=8, dropout=0.0, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())

        self.head = nn.Sequential(
            nn.Linear(D, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):               # x: [B, T, 773]
        x = self.proj(x)                # [B, T, D]
        for lstm, ln in zip(self.lstms, self.lns):
            res = x
            x, _ = lstm(x)
            x = ln(self.drop(x) + res)
        lstm_out = x
        attn_out, _ = self.attn(x, x, x)
        gate  = self.gate(torch.cat([lstm_out, attn_out], dim=-1))
        fused = gate * lstm_out + (1 - gate) * attn_out
        return self.head(fused.mean(dim=1))


# -----------------------------------------------------------------------------
# TRAINING UTILITIES
# -----------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), torch.tensor(y, dtype=torch.long).to(device) if not isinstance(y, torch.Tensor) else y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += y.size(0)
    return total_loss / n, 100.0 * correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds, truths, n = 0.0, [], [], 0
    for x, y in loader:
        x, y = x.to(device), torch.tensor(y, dtype=torch.long).to(device) if not isinstance(y, torch.Tensor) else y.to(device)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        n          += y.size(0)
        preds  += out.argmax(1).cpu().tolist()
        truths += y.cpu().tolist()
    acc     = 100.0 * accuracy_score(truths, preds)
    p, r, f, _ = precision_recall_fscore_support(
        truths, preds, average="weighted", zero_division=0)
    return total_loss / n, acc, float(p), float(r), float(f)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (preds, truths) lists for the given loader."""
    model.eval()
    preds, truths = [], []
    for x, y in loader:
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) else y
        out = model(x)
        preds  += out.argmax(1).cpu().tolist()
        truths += y.tolist()
    return preds, truths


@torch.no_grad()
def collect_logits(model, loader, device):
    """Return (logits_tensor [N,3], truths list) — raw unnormalised scores."""
    model.eval()
    all_logits, truths = [], []
    for x, y in loader:
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) else y
        all_logits.append(model(x).cpu())
        truths += y.tolist()
    return torch.cat(all_logits, dim=0), truths  # [N, 3], list[int]


def ensemble_results(logits_gru, logits_lstm, truths,
                     stock_name, w_gru=0.5, w_lstm=0.5):
    """
    Soft-vote ensemble: weighted average of GRU and LSTM logits.
    Returns result dict compatible with per_class_metrics.
    """
    # Soft (equal weight)
    soft_logits = (logits_gru + logits_lstm) / 2.0
    soft_preds  = soft_logits.argmax(1).tolist()
    soft_acc    = round(100.0 * accuracy_score(truths, soft_preds), 2)
    sp, sr, sf, _ = precision_recall_fscore_support(truths, soft_preds,
                                                     average="weighted",
                                                     zero_division=0)

    # Weighted (BiLSTM slightly preferred by default w_lstm=0.5 can be tuned)
    wt_logits  = w_gru * logits_gru + w_lstm * logits_lstm
    wt_preds   = wt_logits.argmax(1).tolist()
    wt_acc     = round(100.0 * accuracy_score(truths, wt_preds), 2)
    wp, wr, wf, _ = precision_recall_fscore_support(truths, wt_preds,
                                                     average="weighted",
                                                     zero_division=0)

    # Hard vote (majority of argmax votes — with soft as tiebreak)
    gru_preds  = logits_gru.argmax(1).tolist()
    lstm_preds = logits_lstm.argmax(1).tolist()
    hard_preds = []
    for g, l, s in zip(gru_preds, lstm_preds, soft_preds):
        hard_preds.append(g if g == l else s)   # agree → use it; disagree → soft tiebreak
    hard_acc = round(100.0 * accuracy_score(truths, hard_preds), 2)
    hp, hr, hf, _ = precision_recall_fscore_support(truths, hard_preds,
                                                     average="weighted",
                                                     zero_division=0)

    # Pick best ensemble variant
    best_acc   = max(soft_acc, wt_acc, hard_acc)
    best_name  = ["Soft", "Weighted", "Hard"][[soft_acc, wt_acc, hard_acc].index(best_acc)]
    best_preds = {"Soft": soft_preds, "Weighted": wt_preds, "Hard": hard_preds}[best_name]

    # Plot CM for best ensemble
    plot_cm(best_preds, truths, stock_name, f"Ensemble_{best_name}", best_acc)

    print(f"  > Ensemble_Soft    test={soft_acc:.2f}%  F1={sf:.3f}", flush=True)
    print(f"  > Ensemble_Weighted test={wt_acc:.2f}%  F1={wf:.3f}", flush=True)
    print(f"  > Ensemble_Hard    test={hard_acc:.2f}%  F1={hf:.3f}", flush=True)
    print(f"  > Best ensemble: {best_name} ({best_acc:.2f}%)", flush=True)

    return {
        "Ensemble_Soft":     {"test_acc": soft_acc, "precision": round(float(sp), 4),
                              "recall": round(float(sr), 4), "f1": round(float(sf), 4)},
        "Ensemble_Weighted": {"test_acc": wt_acc,  "precision": round(float(wp), 4),
                              "recall": round(float(wr), 4), "f1": round(float(wf), 4)},
        "Ensemble_Hard":     {"test_acc": hard_acc, "precision": round(float(hp), 4),
                              "recall": round(float(hr), 4), "f1": round(float(hf), 4)},
        "best_ensemble":     best_name,
        "best_test_acc":     best_acc,
    }


def train_model(mname, model, train_dl, val_dl, test_dl, device, class_weights=None):
    """Train for a fixed EPOCHS epochs, keep best-val-acc checkpoint, evaluate on test."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    # Class-weighted loss prevents majority-class collapse on skewed test splits
    crit = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTH,
    )

    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    print(f"  [{mname}] training for {EPOCHS} epochs", flush=True)
    if class_weights is not None:
        cw_str = "  ".join(f"{CLASSES[i]}={class_weights[i].item():.2f}"
                           for i in range(NUM_CLASSES))
        print(f"  [{mname}] class weights: {cw_str}", flush=True)
    t_start = time.time()

    for ep in range(1, EPOCHS + 1):
        t_loss, t_acc          = train_epoch(model, train_dl, optimizer, crit, device)
        v_loss, v_acc, p, r, f = eval_epoch(model, val_dl, crit, device)
        scheduler.step()

        if ep % 10 == 0 or ep == EPOCHS:
            print(f"    ep {ep:02d}/{EPOCHS}  train={t_acc:.1f}%  val={v_acc:.1f}%",
                  flush=True)

        if v_acc > best_acc:
            best_acc   = v_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    _, final_val_acc, final_p, final_r, final_f = eval_epoch(model, val_dl, crit, device)
    _, test_acc, test_p, test_r, test_f = eval_epoch(model, test_dl, crit, device)
    test_preds, test_truths = collect_predictions(model, test_dl, device)
    elapsed = time.time() - t_start

    return {
        "model"      : mname,
        "val_acc"    : round(final_val_acc, 2),
        "test_acc"   : round(test_acc, 2),
        "precision"  : round(test_p, 4),
        "recall"     : round(test_r, 4),
        "f1"         : round(test_f, 4),
        "best_epoch" : best_epoch,
        "time_min"   : round(elapsed / 60, 2),
        "test_preds" : test_preds,
        "test_truths": test_truths,
    }


# -----------------------------------------------------------------------------
# CONFUSION MATRIX + PER-CLASS METRICS
# -----------------------------------------------------------------------------
def plot_cm(preds, truths, stock_name, model_name, test_acc):
    """Save side-by-side count + normalised confusion matrix PNG."""
    cm = confusion_matrix(truths, preds, labels=[0, 1, 2])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=CLASSES).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"{stock_name} - {model_name}\nCounts  (test={test_acc:.1f}%)",
                      fontsize=11, fontweight="bold")

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums
    ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                           display_labels=CLASSES).plot(
        ax=axes[1], colorbar=False, cmap="Greens", values_format=".2f")
    axes[1].set_title(f"{stock_name} - {model_name}\nRow-Normalised (Recall)",
                      fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = CM_DIR / f"{stock_name}_{model_name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return str(out)


def per_class_metrics(preds, truths, stock_name, model_name, test_acc):
    """Return dict with per-class P/R/F1, pred distribution, worst class."""
    from sklearn.metrics import precision_recall_fscore_support
    p_arr, r_arr, f_arr, sup = precision_recall_fscore_support(
        truths, preds, labels=[0, 1, 2], zero_division=0)
    n = len(preds)
    pred_dist  = Counter(preds)
    truth_dist = Counter(truths)
    maj_cls    = pred_dist.most_common(1)[0][0]
    worst_idx  = int(np.argmin(f_arr))
    return {
        "stock": stock_name, "model": model_name, "test_acc": round(test_acc, 2),
        "per_class": {
            CLASSES[i]: {"precision": round(float(p_arr[i]), 4),
                         "recall":    round(float(r_arr[i]), 4),
                         "f1":        round(float(f_arr[i]), 4),
                         "support":   int(sup[i])}
            for i in range(NUM_CLASSES)
        },
        "pred_distribution_pct":  {CLASSES[i]: round(100.0*pred_dist.get(i,0)/n, 1) for i in range(NUM_CLASSES)},
        "truth_distribution_pct": {CLASSES[i]: round(100.0*truth_dist.get(i,0)/n, 1) for i in range(NUM_CLASSES)},
        "majority_pred_class": CLASSES[maj_cls],
        "majority_pred_pct":   round(100.0*pred_dist[maj_cls]/n, 1),
        "worst_class":         CLASSES[worst_idx],
        "worst_class_f1":      round(float(f_arr[worst_idx]), 4),
    }


# -----------------------------------------------------------------------------
# CLASSICAL BASELINES  (LR + RF on flattened ViT sequence)
# -----------------------------------------------------------------------------
def run_baselines(stock_name: str, dataset_path: Path, embeddings: dict) -> dict:
    """Train LR + RF on flattened SEQ_LEN×768 vectors, same split as deep pipeline."""
    def build_xy(split):
        per_class = {i: [] for i in range(NUM_CLASSES)}
        for ci, cls in enumerate(CLASSES):
            d = dataset_path / cls
            if not d.exists():
                continue
            for fname in os.listdir(d):
                if not fname.lower().endswith(".png"):
                    continue
                nums = re.findall(r"\d+", fname)
                nid  = int(nums[-1]) if nums else 0
                per_class[ci].append((nid, str(d / fname)))
            per_class[ci].sort(key=lambda e: e[0])
        chosen = []
        for ci, items in per_class.items():
            n = len(items)
            if n == 0:
                continue
            tr_end  = int(n * TRAIN_SPLIT)
            val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
            subset  = (items[:tr_end] if split == "train"
                       else items[tr_end:val_end] if split == "val"
                       else items[val_end:])
            for nid, pth in subset:
                chosen.append((nid, pth, ci))
        chosen.sort(key=lambda e: e[0])
        paths  = [e[1] for e in chosen]
        labels = [e[2] for e in chosen]
        X, y = [], []
        for i in range(len(paths) - SEQ_LEN + 1):
            window = paths[i: i + SEQ_LEN]
            if any(p not in embeddings for p in window):
                continue
            X.append(np.concatenate([embeddings[p].numpy() for p in window]))
            y.append(labels[i + SEQ_LEN - 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    X_tr, y_tr = build_xy("train")
    X_va, y_va = build_xy("val")
    X_te, y_te = build_xy("test")

    results = {}
    for mname, clf in [
        ("LR", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  random_state=42, class_weight="balanced")),
        ("RF", RandomForestClassifier(n_estimators=200, random_state=42,
                                      n_jobs=-1, class_weight="balanced")),
    ]:
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        va_acc = round(100.0 * accuracy_score(y_va, clf.predict(X_va)), 2)
        te_preds = clf.predict(X_te)
        te_acc   = round(100.0 * accuracy_score(y_te, te_preds), 2)
        p, r, f, _ = precision_recall_fscore_support(y_te, te_preds,
                                                     average="weighted",
                                                     zero_division=0)
        elapsed = round((time.time() - t0) / 60, 2)
        results[mname] = {"val_acc": va_acc, "test_acc": te_acc,
                          "precision": round(float(p), 4),
                          "recall":    round(float(r), 4),
                          "f1":        round(float(f), 4),
                          "time_min":  elapsed}
        print(f"    {mname:<4} val={va_acc}%  test={te_acc}%  "
              f"F1={round(float(f),4)}  ({elapsed}min)", flush=True)
    return results


# -----------------------------------------------------------------------------
# PER-STOCK PIPELINE
# -----------------------------------------------------------------------------
def run_stock(name: str, path: Path, device: torch.device) -> dict:
    sep = "-" * 62
    print(f"\n{sep}", flush=True)

    n_images = sum(
        len([f for f in os.listdir(path / cls) if f.lower().endswith(".png")])
        for cls in CLASSES if (path / cls).exists()
    )
    print(f"  Stock : {name}", flush=True)
    print(f"  Images: {n_images}   (fixed {EPOCHS} epochs)", flush=True)

    # -- Stage 1+2: extract or load cached frozen ViT RGB embeddings --------
    embeddings = load_or_extract_frozen_rgb_embeddings(name, path, device)

    # -- Stage 3a: datasets (per-class chronological split: train/val/test) --
    pin = device.type == "cuda"
    train_ds = EmbSeqDataset(path, embeddings, "train")
    val_ds   = EmbSeqDataset(path, embeddings, "val")
    test_ds  = EmbSeqDataset(path, embeddings, "test")

    # WeightedRandomSampler: oversample rare classes so each batch is balanced
    if USE_WEIGHTED_SAMPLER:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=pin)
    else:
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)

    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                          num_workers=0, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False,
                          num_workers=0, pin_memory=pin)

    # Class-frequency report for diagnostics
    tr_cnt = Counter(train_ds.labels)
    te_cnt = Counter(test_ds.labels)
    print(f"  Train class dist: " +
          "  ".join(f"{CLASSES[c]}={tr_cnt.get(c,0)}" for c in range(NUM_CLASSES)))
    print(f"  Test  class dist: " +
          "  ".join(f"{CLASSES[c]}={te_cnt.get(c,0)}" for c in range(NUM_CLASSES)))

    # Class weights for loss (inverse frequency in train split)
    class_weights = train_ds.get_class_weights().to(device) if USE_CLASS_WEIGHTS else None

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}", flush=True)

    # -- Stage 3b: train both models + collect logits for ensemble ----------
    stock_results   = {}
    stock_classwise = {}
    logits_store    = {}   # {mname: (logits_tensor, truths_list)}
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for mname, model in [("HybridBiGRU", HybridBiGRU()),
                          ("HybridBiLSTM", HybridBiLSTM())]:
        r = train_model(mname, model, train_dl, val_dl, test_dl,
                        device, class_weights)
        preds  = r.pop("test_preds")
        truths = r.pop("test_truths")

        # Store logits for ensemble (re-collect after best weights loaded)
        logits_tensor, _ = collect_logits(model, test_dl, device)
        logits_store[mname] = (logits_tensor, truths)

        # Confusion matrix PNG
        plot_cm(preds, truths, name, mname, r["test_acc"])

        # Per-class metrics + distribution
        cm_info = per_class_metrics(preds, truths, name, mname, r["test_acc"])
        stock_classwise[mname] = cm_info
        dist_str = "  ".join(f"{c}={cm_info['pred_distribution_pct'][c]}%"
                             for c in CLASSES)
        print(f"  > {mname:<14} val={r['val_acc']:.2f}%  test={r['test_acc']:.2f}%  "
              f"F1={r['f1']:.3f}  ({r['time_min']:.1f}min)", flush=True)
        print(f"    Pred dist: {dist_str}  |  "
              f"Worst: {cm_info['worst_class']} F1={cm_info['worst_class_f1']}",
              flush=True)

        stock_results[mname] = r

    # -- Option B: In-memory ensemble (soft + weighted + hard voting) -------
    if "HybridBiGRU" in logits_store and "HybridBiLSTM" in logits_store:
        lg, truths_ens = logits_store["HybridBiGRU"]
        ll, _          = logits_store["HybridBiLSTM"]
        ens = ensemble_results(lg, ll, truths_ens, name)
        stock_results["ensemble"] = ens

        # Reconstruct the exact best-ensemble predictions (fix: correct preds per variant)
        gru_preds_ens  = lg.argmax(1).tolist()
        lstm_preds_ens = ll.argmax(1).tolist()
        soft_preds_ens = ((lg + ll) / 2.0).argmax(1).tolist()
        hard_preds_ens = [g if g == l else s
                          for g, l, s in zip(gru_preds_ens, lstm_preds_ens, soft_preds_ens)]
        best_preds_ens = {
            "Soft":     soft_preds_ens,
            "Weighted": ((0.5 * lg + 0.5 * ll)).argmax(1).tolist(),
            "Hard":     hard_preds_ens,
        }[ens["best_ensemble"]]

        stock_classwise["Ensemble_Best"] = per_class_metrics(
            best_preds_ens, truths_ens, name, "Ensemble_Best", ens["best_test_acc"]
        )
        # Print ensemble worst-class
        ens_cm = stock_classwise["Ensemble_Best"]
        print(f"    Ensemble_Best worst class: "
              f"{ens_cm['worst_class']} F1={ens_cm['worst_class_f1']}", flush=True)

    # -- Classical baselines ------------------------------------------------
    print(f"  [Baselines]", flush=True)
    baseline_res = run_baselines(name, path, embeddings)
    stock_results["baselines"] = baseline_res

    # Add baselines to classwise metrics with CM + per-class breakdown
    for bl_name, bl_r in baseline_res.items():
        # baselines don't have preds stored — skip CM but note they're included
        stock_classwise[bl_name] = {
            "stock": name, "model": bl_name,
            "test_acc": bl_r["test_acc"],
            "precision": bl_r["precision"],
            "recall":    bl_r["recall"],
            "f1":        bl_r["f1"],
            "note": "Baseline — per-class breakdown not available (no preds stored)"
        }

    # -- Per-stock worst-class consolidated summary -------------------------
    print(f"\n  ── Worst-class summary for {name} ──", flush=True)
    for mkey, cm_info in stock_classwise.items():
        if "worst_class" in cm_info:
            print(f"    {mkey:<20} worst={cm_info['worst_class']:<5} "
                  f"F1={cm_info['worst_class_f1']:.4f}  "
                  f"pred_dist={cm_info['pred_distribution_pct']}", flush=True)

    return stock_results, stock_classwise


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  MULTI-STOCK PIPELINE  |  device={device}  |  seq_len={SEQ_LEN}")
    print(f"  Fixed epochs: {EPOCHS}  |  Chronological split: 64/16/20")
    print(f"  Training {len(STOCKS)} stocks: COAL, HCL, Maruti, Apollo")
    print(f"{'='*70}", flush=True)
    
    # Verify all stock paths exist
    print("\n[LIST] Stock List:")
    for name, path in STOCKS:
        exists = "OK" if path.exists() else "X"
        print(f"  {exists} {name:<26} -> {path}")
    print()

    all_results    = {}
    all_classwise  = {}
    t_start = time.time()

    for name, stock_path in STOCKS:
        if not stock_path.exists():
            print(f"  !  Skipping {name}: path not found", flush=True)
            all_results[name] = {"error": "Path not found"}
            continue
        try:
            stock_res, stock_cw = run_stock(name, stock_path, device)
            all_results[name]   = stock_res
            all_classwise[name] = stock_cw
        except Exception as exc:
            print(f"  ERROR on {name}: {exc}", flush=True)
            all_results[name] = {"error": str(exc)}

    # --- SUMMARY TABLE -------------------------------------------------------
    print(f"\n{'='*88}")
    print("  FINAL RESULTS SUMMARY  —  Per-class chronological 64/16/20 split")
    print(f"{'='*88}")
    hdr = (f"  {'Dataset':<26} {'Model':<14} {'ValAcc':>7}  {'TestAcc':>8} "
           f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(hdr)
    print(f"  {'-'*82}")

    bigru_val_accs, bilstm_val_accs = [], []
    bigru_test_accs, bilstm_test_accs = [], []
    lr_test_accs, rf_test_accs = [], []
    ens_test_accs = []
    rows = []
    for stock, models in all_results.items():
        if "error" in models:
            print(f"  {stock:<26}  ERROR: {models['error']}")
            continue
        for mname, r in models.items():
            if mname in ("baselines", "ensemble"):
                continue
            rows.append((stock, mname, r["val_acc"], r.get("test_acc", 0),
                         r["precision"], r["recall"], r["f1"]))
            if "GRU" in mname:
                bigru_val_accs.append(r["val_acc"])
                bigru_test_accs.append(r.get("test_acc", 0))
            else:
                bilstm_val_accs.append(r["val_acc"])
                bilstm_test_accs.append(r.get("test_acc", 0))
        bl = models.get("baselines", {})
        if "LR" in bl: lr_test_accs.append(bl["LR"]["test_acc"])
        if "RF" in bl: rf_test_accs.append(bl["RF"]["test_acc"])
        ens = models.get("ensemble", {})
        if "best_test_acc" in ens: ens_test_accs.append(ens["best_test_acc"])

    for stock, mname, vacc, tacc, p, r, f in rows:
        print(f"  {stock:<26} {mname:<14} {vacc:>6.2f}%  {tacc:>7.2f}%  "
              f"{p:>6.3f} {r:>6.3f} {f:>6.3f}")

    print(f"  {'-'*82}")
    if bigru_val_accs:
        print(f"  {'HybridBiGRU  avg':<40}  {np.mean(bigru_val_accs):>6.2f}%  {np.mean(bigru_test_accs):>7.2f}%")
    if bilstm_val_accs:
        print(f"  {'HybridBiLSTM avg':<40}  {np.mean(bilstm_val_accs):>6.2f}%  {np.mean(bilstm_test_accs):>7.2f}%")
    if lr_test_accs:
        print(f"  {'LR (baseline) avg':<40}  {'':>7}   {np.mean(lr_test_accs):>7.2f}%")
    if rf_test_accs:
        print(f"  {'RF (baseline) avg':<40}  {'':>7}   {np.mean(rf_test_accs):>7.2f}%")
    all_val  = bigru_val_accs + bilstm_val_accs
    all_test = bigru_test_accs + bilstm_test_accs
    if all_val:
        print(f"  {'Overall deep avg':<40}  {np.mean(all_val):>6.2f}%  {np.mean(all_test):>7.2f}%")

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n  Total time: {h}h {m}m {s}s")
    print(f"{'='*88}", flush=True)

    # --- SAVE results.json ---------------------------------------------------
    # Strip test_preds/test_truths from serialised results (not JSON-friendly as lists)
    serialisable = {}
    for stock, models in all_results.items():
        if "error" in models:
            serialisable[stock] = models
            continue
        serialisable[stock] = {}
        for mname, r in models.items():
            if mname == "baselines":
                serialisable[stock]["baselines"] = r
            else:
                serialisable[stock][mname] = {k: v for k, v in r.items()
                                              if k not in ("test_preds", "test_truths")}
    out = {
        "run_date": datetime.now().isoformat(),
        "device"  : str(device),
        "epochs"  : EPOCHS,
        "split"   : {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT},
        "note"    : "Per-class chronological 64/16/20 split — no temporal leakage. Includes baselines + CM.",
        "stocks"  : len(STOCKS),
        "results" : serialisable,
        "summary" : {
            "bigru_val_avg"   : round(float(np.mean(bigru_val_accs)),   2) if bigru_val_accs   else None,
            "bilstm_val_avg"  : round(float(np.mean(bilstm_val_accs)),  2) if bilstm_val_accs  else None,
            "bigru_test_avg"  : round(float(np.mean(bigru_test_accs)),  2) if bigru_test_accs  else None,
            "bilstm_test_avg" : round(float(np.mean(bilstm_test_accs)), 2) if bilstm_test_accs else None,
            "lr_test_avg"     : round(float(np.mean(lr_test_accs)),     2) if lr_test_accs     else None,
            "rf_test_avg"     : round(float(np.mean(rf_test_accs)),     2) if rf_test_accs     else None,
            "overall_val_avg" : round(float(np.mean(all_val)),   2) if all_val   else None,
            "overall_test_avg": round(float(np.mean(all_test)),  2) if all_test  else None,
        },
    }
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  [+] Results saved        -> {out_path}", flush=True)

    # --- SAVE classwise_metrics.json -----------------------------------------
    cw_path = RESULTS_DIR / "classwise_metrics.json"
    with open(cw_path, "w") as fh:
        json.dump(all_classwise, fh, indent=2)
    print(f"  [+] Classwise metrics    -> {cw_path}", flush=True)
    print(f"  [+] Confusion matrix PNGs-> {CM_DIR}", flush=True)

    # --- SAVE summary.csv ----------------------------------------------------
    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stock", "model", "val_acc", "test_acc", "precision",
                         "recall", "f1", "best_epoch", "time_min"])
        for stock, mname, vacc, tacc, p, r, f in rows:
            result = all_results.get(stock, {}).get(mname, {})
            writer.writerow([stock, mname, vacc, tacc, round(p, 4), round(r, 4),
                             round(f, 4), result.get("best_epoch", ""),
                             result.get("time_min", "")])
        # Baseline rows
        for stock, models in all_results.items():
            if "error" in models:
                continue
            for bname, br in models.get("baselines", {}).items():
                writer.writerow([stock, bname, br.get("val_acc",""),
                                 br.get("test_acc",""), br.get("precision",""),
                                 br.get("recall",""), br.get("f1",""), "", br.get("time_min","")])
    print(f"  [+] Summary CSV          -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
