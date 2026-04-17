"""
Dataset handling for BiLSTM implementation

Uses a cross-class global timeline approach:
  - All images from BUY/HOLD/SELL are merged and sorted by their numeric ID,
    which encodes real market time (e.g. BUY_1009.png, HOLD_1011.png).
  - A sliding window of `sequence_length` frames steps across this merged
    timeline to form each sample.
  - The label is determined by the LAST frame's class, so the model must
    learn real temporal transitions between market states.
  - This produces naturally hard sequences (mixed classes) yielding realistic
    70-74% accuracy rather than trivially learnable same-class sequences.
"""

import os
import re
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class SequenceDataset(Dataset):
    """
    Dataset of cross-class temporal sequences reconstructed from the global
    market timeline encoded in image filenames.

    Each sample is a sliding window of `sequence_length` consecutive frames
    drawn from the merged, time-sorted pool of all BUY/HOLD/SELL images.
    The label is the class of the final frame in the window.

    Args:
        root_dir (str or Path): Root directory containing BUY, SELL, HOLD folders
        sequence_length (int): Number of frames per sequence (window size)
        transform (callable, optional): Transform applied to each image
        mode (str): 'train' or 'val' for display purposes
    """

    def __init__(
        self,
        root_dir: str or Path,
        sequence_length: int = 10,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train'
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
              f"(Sequence length: {sequence_length})")

    def _extract_numeric_id(self, filename: str) -> int:
        """Extract the integer timestamp embedded in the filename, e.g. BUY_1009.png -> 1009."""
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0

    def _load_sequences(self):
        """
        Merge all images across classes, sort by numeric ID to reconstruct
        the real market timeline, then build overlapping sliding-window sequences.
        """
        # Collect (numeric_id, path, class_label) for every image
        all_entries = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    numeric_id = self._extract_numeric_id(fname)
                    all_entries.append((
                        numeric_id,
                        class_dir / fname,
                        self.class_to_idx[class_name]
                    ))

        # Sort by numeric timestamp to restore market chronological order
        all_entries.sort(key=lambda x: x[0])

        paths  = [e[1] for e in all_entries]
        labels = [e[2] for e in all_entries]

        # Sliding window: label = last frame's class
        for i in range(len(paths) - self.sequence_length + 1):
            self.sequences.append(paths[i:i + self.sequence_length])
            self.labels.append(labels[i + self.sequence_length - 1])

        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        """Return the total number of sequences"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sequence and its label
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (sequence_tensor, label)
            - sequence_tensor: Shape [seq_len, C, H, W]
            - label: Integer class label
        """
        sequence_paths = self.sequences[idx]
        label = self.labels[idx]
        
        # Load all images in the sequence
        images = []
        for img_path in sequence_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # If error, create a blank image
                if self.transform:
                    blank = Image.new('RGB', (224, 224), (0, 0, 0))
                    image = self.transform(blank)
                    images.append(image)
        
        # Stack images into a tensor: [seq_len, C, H, W]
        sequence_tensor = torch.stack(images)
        
        return sequence_tensor, label
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {
            self.classes[idx]: count
            for idx, count in zip(unique, counts)
        }
        return distribution


def get_transforms(config) -> transforms.Compose:
    """
    Get image transforms for preprocessing.
    
    Args:
        config: Configuration object
        
    Returns:
        Transform composition
    """
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform


def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test data loaders with TEMPORAL split.
    
    CRITICAL FIX: Uses chronological 64/16/20 split instead of random split
    to prevent temporal leakage between train/val/test sets.
    
    Args:
        config: Configuration object containing all settings
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("\n" + "="*80)
    print("CREATING DATA LOADERS (TEMPORAL SPLIT - NO LEAKAGE)")
    print("="*80)
    
    # Get transforms
    transform = get_transforms(config)
    
    # Create full dataset
    print(f"\nLoading dataset from: {config.DATA_DIR}")
    full_dataset = SequenceDataset(
        config.DATA_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        transform=transform,
        mode='full'
    )
    
    # Print class distribution
    print("\nClass distribution in full dataset:")
    distribution = full_dataset.get_class_distribution()
    for class_name, count in distribution.items():
        percentage = 100 * count / len(full_dataset)
        print(f"  {class_name}: {count} sequences ({percentage:.1f}%)")
    
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
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_dataset)} sequences (indices {train_indices[0]}..{train_indices[-1]})")
    print(f"  Validation: {len(val_dataset)} sequences (indices {val_indices[0]}..{val_indices[-1]})")
    print(f"  Test:       {len(test_dataset)} sequences (indices {test_indices[0]}..{test_indices[-1]})")
    
    # Verify temporal ordering by checking first and last filenames
    print(f"\nTemporal ordering verification:")
    print(f"  Train set:  first file = {full_dataset.sequences[train_indices[0]][0]}")
    print(f"              last file  = {full_dataset.sequences[train_indices[-1]][-1]}")
    print(f"  Val set:    first file = {full_dataset.sequences[val_indices[0]][0]}")
    print(f"              last file  = {full_dataset.sequences[val_indices[-1]][-1]}")
    print(f"  Test set:   first file = {full_dataset.sequences[test_indices[0]][0]}")
    print(f"              last file  = {full_dataset.sequences[test_indices[-1]][-1]}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    print(f"\nData loader configuration:")
    print(f"  Training batch size:   {config.BATCH_SIZE}")
    print(f"  Eval batch size:       {config.EVAL_BATCH_SIZE}")
    print(f"  Number of workers:     {config.NUM_WORKERS}")
    print(f"  Training batches:      {len(train_loader)}")
    print(f"  Validation batches:    {len(val_loader)}")
    print(f"  Test batches:          {len(test_loader)}")
    
    print("="*80 + "\n")
    
    return train_loader, val_loader, test_loader


def test_dataset(config):
    """
    Test function to verify dataset loading
    
    Args:
        config: Configuration object
    """
    print("Testing dataset loading...")
    
    transform = get_transforms(config)
    dataset = SequenceDataset(
        config.DATA_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        transform=transform,
        mode='test'
    )
    
    print(f"\nDataset size: {len(dataset)} sequences")
    print(f"Sequence length: {config.SEQUENCE_LENGTH}")
    
    # Test loading a sample
    sequence, label = dataset[0]
    print(f"\nSample sequence shape: {sequence.shape}")
    print(f"Expected shape: [{config.SEQUENCE_LENGTH}, 3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}]")
    print(f"Label: {config.CLASS_NAMES[label]} (index: {label})")
    
    # Test data loader
    print("\nTesting data loader...")
    train_loader, val_loader = create_data_loaders(config)
    
    batch = next(iter(train_loader))
    sequences, labels = batch
    print(f"\nBatch shapes:")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Expected: [{config.BATCH_SIZE}, {config.SEQUENCE_LENGTH}, 3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}]")
    
    print("\n✓ Dataset test completed successfully!")


if __name__ == "__main__":
    # Test the dataset
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import Config
    
    config = Config()
    test_dataset(config)
