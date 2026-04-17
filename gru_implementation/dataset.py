"""
Dataset handling for BiGRU implementation.

Uses the same cross-class global timeline approach as the BiLSTM implementation:
  - All BUY/HOLD/SELL images are merged and sorted by their embedded numeric ID,
    which encodes real market time (e.g. BUY_1009.png, HOLD_1011.png).
  - A sliding window of `sequence_length` frames steps across the merged
    timeline to form each sample.
  - The label is the class of the LAST frame, so the model must learn real
    temporal transitions between market states.
"""

import os
import re
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class SequenceDataset(Dataset):
    """
    Dataset of cross-class temporal sequences from the global market timeline.

    Args:
        root_dir: Root directory containing BUY, SELL, HOLD sub-folders.
        sequence_length: Sliding window size (number of frames per sample).
        transform: Optional torchvision transform applied to each image.
        mode: Display label ('train', 'val', 'full', …).
    """

    def __init__(
        self,
        root_dir,
        sequence_length: int = 10,
        transform=None,
        mode: str = 'train',
    ):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.mode = mode

        self.classes = ['BUY', 'HOLD', 'SELL']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.sequences = []
        self.labels = []
        self._load_sequences()

        print(f"{mode.capitalize()} dataset: {len(self.sequences)} sequences "
              f"(window={sequence_length})")

    # ------------------------------------------------------------------
    def _extract_numeric_id(self, filename: str) -> int:
        nums = re.findall(r'\d+', filename)
        return int(nums[-1]) if nums else 0

    def _load_sequences(self):
        all_entries = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                print(f"Warning: {cls_dir} does not exist")
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    nid = self._extract_numeric_id(fname)
                    all_entries.append((nid, cls_dir / fname, self.class_to_idx[cls]))

        all_entries.sort(key=lambda e: e[0])

        paths  = [e[1] for e in all_entries]
        labels = [e[2] for e in all_entries]

        for i in range(len(paths) - self.sequence_length + 1):
            self.sequences.append(paths[i: i + self.sequence_length])
            self.labels.append(labels[i + self.sequence_length - 1])

        self.labels = np.array(self.labels)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        paths = self.sequences[idx]
        label = int(self.labels[idx])

        images = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                blank = Image.new('RGB', (224, 224), (0, 0, 0))
                images.append(self.transform(blank) if self.transform else blank)

        return torch.stack(images), label   # [T, C, H, W], int

    # ------------------------------------------------------------------
    def get_class_distribution(self) -> dict:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {self.classes[idx]: int(cnt) for idx, cnt in zip(unique, counts)}


# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders with TEMPORAL split.
    
    CRITICAL FIX: Uses chronological 64/16/20 split instead of random split
    to prevent temporal leakage between train/val/test sets.

    Args:
        config: Config instance.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("\n" + "=" * 80)
    print("CREATING DATA LOADERS (TEMPORAL SPLIT - NO LEAKAGE)")
    print("=" * 80)

    transform = get_transforms(config)

    print(f"\nLoading dataset from: {config.DATA_DIR}")
    full_dataset = SequenceDataset(
        config.DATA_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        transform=transform,
        mode='full',
    )

    print("\nClass distribution:")
    for cls, cnt in full_dataset.get_class_distribution().items():
        pct = 100 * cnt / len(full_dataset)
        print(f"  {cls}: {cnt} ({pct:.1f}%)")

    # TEMPORAL SPLIT (64% train / 16% val / 20% test) - NO SHUFFLING
    print(f"\n{'='*80}")
    print("APPLYING TEMPORAL SPLIT (NO RANDOM SHUFFLE)")
    print(f"{'='*80}")
    print("Split ratios: 64% train / 16% validation / 20% test")
    print("⚠️  This preserves chronological order to prevent temporal leakage")
    
    total_sequences = len(full_dataset)
    
    # Safety check for empty dataset
    if total_sequences == 0:
        raise ValueError(f"Dataset is empty! Check that data exists in: {config.DATA_DIR}")
    
    train_end_idx = int(total_sequences * 0.64)
    val_end_idx = int(total_sequences * 0.80)
    
    # Create indices for each split (chronological order preserved)
    train_indices = list(range(0, train_end_idx))
    val_indices = list(range(train_end_idx, val_end_idx))
    test_indices = list(range(val_end_idx, total_sequences))
    
    # Create subset datasets
    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)
    test_ds = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_ds)} sequences (indices {train_indices[0]}..{train_indices[-1]})")
    print(f"  Validation: {len(val_ds)} sequences (indices {val_indices[0]}..{val_indices[-1]})")
    print(f"  Test:       {len(test_ds)} sequences (indices {test_indices[0]}..{test_indices[-1]})")
    
    # Verify temporal ordering by checking first and last filenames
    print(f"\nTemporal ordering verification:")
    print(f"  Train set:  first file = {full_dataset.sequences[train_indices[0]][0]}")
    print(f"              last file  = {full_dataset.sequences[train_indices[-1]][-1]}")
    print(f"  Val set:    first file = {full_dataset.sequences[val_indices[0]][0]}")
    print(f"              last file  = {full_dataset.sequences[val_indices[-1]][-1]}")
    print(f"  Test set:   first file = {full_dataset.sequences[test_indices[0]][0]}")
    print(f"              last file  = {full_dataset.sequences[test_indices[-1]][-1]}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
    )

    print(f"\nData loader configuration:")
    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")
    print(f"  Test batches:    {len(test_loader)}")
    print("=" * 80 + "\n")

    return train_loader, val_loader, test_loader
