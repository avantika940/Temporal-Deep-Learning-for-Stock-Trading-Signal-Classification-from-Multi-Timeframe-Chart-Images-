"""
ViT Fine-tuning on Adani MTF dataset + Embedding extraction for all 11 stocks.

Workflow:
  1. Fine-tune ViT-B/16 on Adani (data/Adani_MTF_Images_224x224/)
       - Freeze all layers except last 4 transformer blocks + head
       - 30 epochs, lr=1e-5, wd=1e-4
       - Augmentation: random horizontal/vertical flip, color jitter, random rotation ±15°
  2. Save fine-tuned model to embeddings/vit_finetuned_mtf.pth
  3. Re-extract CLS-token embeddings for all 11 stocks using fine-tuned ViT
  4. Save per-stock embeddings to embeddings/{stock_name}_finetuned.npz
  5. Print before/after silhouette score comparison
"""

import os
import re
import time
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
import timm
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────── Config ───────────────────────────
ADANI_DIR       = Path(r"d:\MTECH\data\Adani_MTF_Images_224x224")
MULTI_STOCK_DIR = Path(r"d:\MTECH\data_multi_processed")
EMBED_DIR       = Path(r"d:\MTECH\embeddings")
EMBED_DIR.mkdir(exist_ok=True)

MODEL_SAVE_PATH = EMBED_DIR / "vit_finetuned_mtf.pth"
RESULTS_PATH    = EMBED_DIR / "finetuning_results.json"

EPOCHS     = 30
LR         = 1e-5
WD         = 1e-4
BATCH_SIZE = 32
IMG_SIZE   = 224
NUM_BLOCKS_UNFREEZE = 4   # unfreeze last N transformer blocks
SEED       = 42

CLASS_NAMES = ["BUY", "HOLD", "SELL"]
LABEL2IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────── Transforms ───────────────────────────
TRAIN_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ─────────────────────────── Dataset ───────────────────────────
class MTFImageDataset(Dataset):
    """Loads images from BUY/HOLD/SELL sub-folders."""

    def __init__(self, root: Path, transform=None, split="all",
                 train_ratio=0.80, val_ratio=0.10):
        self.transform = transform
        self.samples   = []   # (path, label)

        for cls in CLASS_NAMES:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            files = sorted(cls_dir.glob("*.png"),
                           key=lambda p: self._sort_key(p.stem))
            n = len(files)
            if n == 0:
                continue

            # chronological split per class
            t1 = int(n * train_ratio)
            t2 = int(n * (train_ratio + val_ratio))
            if split == "train":
                chosen = files[:t1]
            elif split == "val":
                chosen = files[t1:t2]
            elif split == "test":
                chosen = files[t2:]
            else:                   # "all"
                chosen = files

            for f in chosen:
                self.samples.append((f, LABEL2IDX[cls]))

    @staticmethod
    def _sort_key(stem: str) -> int:
        m = re.search(r"(\d+)$", stem)
        return int(m.group(1)) if m else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_weighted_sampler(dataset: MTFImageDataset) -> WeightedRandomSampler:
    labels = [lbl for _, lbl in dataset.samples]
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    weights_per_class = 1.0 / (counts + 1e-6)
    sample_weights = torch.tensor([weights_per_class[l] for l in labels])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


# ─────────────────────────── Model ───────────────────────────
def build_model(num_classes: int = 3, unfreeze_blocks: int = NUM_BLOCKS_UNFREEZE):
    """Load pretrained ViT-B/16 via timm, freeze all but last N blocks + head."""
    model = timm.create_model("vit_base_patch16_224", pretrained=True,
                              num_classes=num_classes)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classification head
    for param in model.head.parameters():
        param.requires_grad = True

    # Unfreeze last N transformer blocks
    # timm ViT-B/16 has model.blocks (list of 12 Block modules)
    total_blocks = len(model.blocks)
    start_block  = max(0, total_blocks - unfreeze_blocks)
    for blk in model.blocks[start_block:]:
        for param in blk.parameters():
            param.requires_grad = True

    # Also unfreeze the final LayerNorm
    for param in model.norm.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  "
          f"(unfrozen blocks {start_block}–{total_blocks-1} + head + norm)")
    return model


# ─────────────────────────── Training ───────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


def fine_tune_vit():
    print("\n" + "="*65)
    print("  PHASE 1 – Fine-tuning ViT-B/16 on Adani MTF")
    print("="*65)

    train_ds = MTFImageDataset(ADANI_DIR, TRAIN_TRANSFORM, split="train")
    val_ds   = MTFImageDataset(ADANI_DIR, EVAL_TRANSFORM,  split="val")
    test_ds  = MTFImageDataset(ADANI_DIR, EVAL_TRANSFORM,  split="test")

    # Print class distribution per split
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        lbls = [l for _, l in ds.samples]
        counts = np.bincount(lbls, minlength=3)
        print(f"  {name:5s}: {len(ds):4d} images  "
              f"BUY={counts[0]}  HOLD={counts[1]}  SELL={counts[2]}")

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WD
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc  = 0.0
    best_state    = None
    history       = []

    print(f"\n  Training for {EPOCHS} epochs …\n")
    print(f"  {'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'LR':>10}")
    print("  " + "-"*55)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          optimizer, criterion, epoch)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        cur_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"  {epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.4f}  {cur_lr:>10.2e}  "
              f"({elapsed:.1f}s)")

        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            va_loss=va_loss, va_acc=va_acc))

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            print(f"    ✓ New best val acc: {best_val_acc:.4f}")

    # Restore best weights
    model.load_state_dict(best_state)
    model.to(DEVICE)

    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"\n  Best val acc : {best_val_acc:.4f}")
    print(f"  Test  acc    : {te_acc:.4f}")

    # Save model
    torch.save({"model_state_dict": best_state,
                "num_classes": 3,
                "class_names": CLASS_NAMES,
                "history": history,
                "best_val_acc": best_val_acc,
                "test_acc": te_acc},
               MODEL_SAVE_PATH)
    print(f"\n  Model saved → {MODEL_SAVE_PATH}")
    return model, history, best_val_acc, te_acc


# ─────────────────────────── Embedding extraction ───────────────────────────
def get_cls_extractor(model):
    """Return a function that extracts CLS token from ViT (no classification head)."""
    class CLSExtractor(nn.Module):
        def __init__(self, vit):
            super().__init__()
            self.vit = vit

        def forward(self, x):
            # timm ViT forward_features returns the full feature map;
            # for vit_base_patch16_224 the [CLS] token is at index 0
            feats = self.vit.forward_features(x)   # (B, N+1, D) or (B, D)
            if feats.dim() == 3:
                return feats[:, 0, :]               # CLS token
            return feats                            # already pooled (global avg)

    return CLSExtractor(model).to(DEVICE).eval()


@torch.no_grad()
def extract_embeddings_from_dir(extractor, root: Path,
                                 transform=None) -> tuple:
    """Extract CLS embeddings from all images in BUY/HOLD/SELL sub-dirs."""
    if transform is None:
        transform = EVAL_TRANSFORM

    all_embs, all_labels, all_paths = [], [], []
    for cls in CLASS_NAMES:
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        files = sorted(cls_dir.glob("*.png"))
        for fpath in files:
            img = Image.open(fpath).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)
            emb = extractor(img).cpu().numpy().squeeze()
            all_embs.append(emb)
            all_labels.append(LABEL2IDX[cls])
            all_paths.append(str(fpath))

    return (np.array(all_embs),
            np.array(all_labels),
            np.array(all_paths))


# ─────────────────────────── Silhouette scoring ───────────────────────────
def silhouette_for_npz(npz_path: Path) -> float | None:
    """Compute silhouette score from a saved .npz file."""
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    embs   = data["embeddings"] if "embeddings" in data else data["arr_0"]
    labels = data["labels"]     if "labels"     in data else data["arr_1"]
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(silhouette_score(embs, labels, sample_size=min(2000, len(labels)),
                                      random_state=SEED))
    except Exception:
        return None


# ─────────────────────────── Main ───────────────────────────
def main():
    results = {}

    # ── Phase 1: fine-tune ─────────────────────────────────────────────────
    model, history, best_val_acc, test_acc = fine_tune_vit()
    results["adani_best_val_acc"] = best_val_acc
    results["adani_test_acc"]     = test_acc

    # ── Phase 2: build extractor ───────────────────────────────────────────
    extractor = get_cls_extractor(model)

    # ── Phase 3: compute BEFORE silhouette scores from existing .npz ──────
    print("\n" + "="*65)
    print("  PHASE 2 – Before/After Silhouette Score Comparison")
    print("="*65)

    before_scores = {}
    existing_npz = {
        "vit_finetuned (old)": EMBED_DIR / "vit_embeddings_finetuned.npz",
        "vit_grayscale"       : EMBED_DIR / "vit_embeddings_grayscale.npz",
        "vit_rgb"             : EMBED_DIR / "vit_embeddings_rgb.npz",
    }
    for name, path in existing_npz.items():
        sc = silhouette_for_npz(path)
        before_scores[name] = sc
        print(f"  BEFORE [{name:>22s}]: "
              f"{sc:.4f}" if sc is not None else f"  BEFORE [{name:>22s}]: N/A")

    results["before_silhouette"] = before_scores

    # ── Phase 4: extract new embeddings for all 11 stocks ─────────────────
    print("\n" + "="*65)
    print("  PHASE 3 – Extracting Fine-tuned Embeddings (all 11 stocks)")
    print("="*65)

    all_stocks = {
        "Adani":             ADANI_DIR,
        "Bajaj_Finance":     MULTI_STOCK_DIR / "Bajaj_Finance",
        "BPCL":              MULTI_STOCK_DIR / "BPCL_MTF",
        "Britannia":         MULTI_STOCK_DIR / "britania_MTF",
        "COAL_India":        MULTI_STOCK_DIR / "COAL_India_MTF",
        "HCL":               MULTI_STOCK_DIR / "HCL_MTF",
        "HDFC":              MULTI_STOCK_DIR / "HDFC_MTF",
        "Hindustan_Unilever":MULTI_STOCK_DIR / "Hindustan_Unilever_MTF",
        "ICICI_Bank":        MULTI_STOCK_DIR / "ICICI_Bank_MTF",
        "Indusind_Bank":     MULTI_STOCK_DIR / "Indusind_Bank_MTF",
        "KOTAK_BANK":        MULTI_STOCK_DIR / "KOTAK_BANK_MTF",
    }

    after_scores    = {}
    per_stock_info  = {}

    for stock_name, stock_dir in all_stocks.items():
        print(f"\n  [{stock_name}] → {stock_dir.name}")
        t0 = time.time()

        embs, labels, paths = extract_embeddings_from_dir(extractor, stock_dir)
        elapsed = time.time() - t0

        save_path = EMBED_DIR / f"{stock_name}_finetuned.npz"
        np.savez_compressed(save_path,
                            embeddings=embs,
                            labels=labels,
                            paths=paths,
                            class_names=np.array(CLASS_NAMES))

        # Silhouette score
        if len(np.unique(labels)) >= 2:
            sc = float(silhouette_score(embs, labels,
                                        sample_size=min(2000, len(embs)),
                                        random_state=SEED))
        else:
            sc = None

        counts = np.bincount(labels, minlength=3)
        print(f"    Total: {len(embs):5d}  BUY={counts[0]} HOLD={counts[1]} SELL={counts[2]}")
        print(f"    Embedding dim : {embs.shape[1]}")
        print(f"    Silhouette    : {sc:.4f}" if sc is not None else "    Silhouette    : N/A")
        print(f"    Saved → {save_path.name}  ({elapsed:.1f}s)")

        after_scores[stock_name]   = sc
        per_stock_info[stock_name] = {
            "n_total": int(len(embs)),
            "n_buy":   int(counts[0]),
            "n_hold":  int(counts[1]),
            "n_sell":  int(counts[2]),
            "emb_dim": int(embs.shape[1]),
            "silhouette": sc,
            "saved_to": str(save_path),
        }

    results["after_silhouette"]  = after_scores
    results["per_stock_info"]    = per_stock_info

    # ── Phase 5: Summary ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL SUMMARY")
    print("="*65)
    print(f"\n  Adani fine-tune  →  Best Val Acc: {best_val_acc:.4f}  |  "
          f"Test Acc: {test_acc:.4f}")

    print("\n  BEFORE silhouette scores (existing embeddings):")
    for name, sc in before_scores.items():
        print(f"    {name:>26s}: "
              + (f"{sc:.4f}" if sc is not None else "N/A"))

    print("\n  AFTER silhouette scores (fine-tuned ViT embeddings):")
    for name, sc in after_scores.items():
        print(f"    {name:>26s}: "
              + (f"{sc:.4f}" if sc is not None else "N/A"))

    # Compute mean for comparison
    before_valid = [v for v in before_scores.values() if v is not None]
    after_valid  = [v for v in after_scores.values()  if v is not None]
    if before_valid:
        print(f"\n  Mean BEFORE (existing): {np.mean(before_valid):.4f}")
    if after_valid:
        print(f"  Mean AFTER  (new)     : {np.mean(after_valid):.4f}")

    delta = np.mean(after_valid) - np.mean(before_valid) if (before_valid and after_valid) else None
    if delta is not None:
        arrow = "▲" if delta > 0 else "▼"
        print(f"  Δ silhouette          : {arrow}{abs(delta):.4f}")

    results["mean_before_silhouette"] = float(np.mean(before_valid)) if before_valid else None
    results["mean_after_silhouette"]  = float(np.mean(after_valid))  if after_valid  else None

    # Save JSON results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {RESULTS_PATH}")
    print("\n  Done ✓")


if __name__ == "__main__":
    main()
