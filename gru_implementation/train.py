"""
Training script for BiGRU models.

Handles:
  - Model initialisation
  - Training loop with gradient clipping
  - Validation
  - Early stopping
  - Model checkpointing
  - JSON history logging
"""

import sys
import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from dataset import create_data_loaders
from models import create_model


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    running_loss = total = correct = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for bi, (seqs, labels) in enumerate(pbar):
        seqs, labels = seqs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(seqs)
        loss = criterion(out, labels)
        loss.backward()

        if config.GRADIENT_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)

        optimizer.step()

        _, pred = torch.max(out, 1)
        total   += labels.size(0)
        correct += (pred == labels).sum().item()
        running_loss += loss.item()

        pbar.set_postfix(loss=f'{running_loss/(bi+1):.4f}',
                         acc=f'{100*correct/total:.2f}%')

    return running_loss / len(loader), 100 * correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = total = correct = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for seqs, labels in pbar:
            seqs, labels = seqs.to(device), labels.to(device)

            out  = model(seqs)
            loss = criterion(out, labels)

            _, pred = torch.max(out, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()
            running_loss += loss.item()

    return running_loss / len(loader), 100 * correct / total


def init_csv_logging(csv_path):
    """Initialize CSV file with headers"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])


def log_epoch_to_csv(csv_path, epoch, train_loss, train_acc, val_loss, val_acc):
    """Append epoch metrics to CSV file"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_model(config):
    """
    Full training pipeline for one BiGRU model variant.

    Args:
        config: Config instance.

    Returns:
        (model, history, best_val_acc)
    """
    config.print_config()
    
    # Validate paths
    config.validate_paths()
    
    set_seed(config.RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.DEVICE = str(device)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = create_data_loaders(config)

    # Model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    model = create_model(config).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # Optimizer
    opt_map = {
        'adam':  optim.Adam,
        'adamw': optim.AdamW,
    }
    if config.OPTIMIZER.lower() in opt_map:
        optimizer = opt_map[config.OPTIMIZER.lower()](
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )

    # Scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE,
                min_lr=config.SCHEDULER_MIN_LR,
            )
        elif config.SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
            )
        elif config.SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA
            )

    history = {'train_loss': [], 'train_acc': [],
               'val_loss':   [], 'val_acc':   [], 'learning_rates': []}

    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0
    t0             = time.time()

    # Initialize CSV logging
    csv_path = config.LOGS_DIR / f"training_curve_{config.MODEL_TYPE}_bigru.csv"
    init_csv_logging(csv_path)

    print(f"\n{'='*80}")
    print(f"TRAINING {config.MODEL_TYPE.upper()} BiGRU  —  {config.NUM_EPOCHS} epochs")
    print(f"{'='*80}\n")

    for epoch in range(config.NUM_EPOCHS):
        t_ep = time.time()
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 80)

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, config)
        vl_loss, vl_acc = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Log to CSV
        log_epoch_to_csv(csv_path, epoch + 1, tr_loss, tr_acc, vl_loss, vl_acc)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
        print(f"  Val   Loss: {vl_loss:.4f} | Val   Acc: {vl_acc:.2f}%")
        print(f"  Time: {time.time()-t_ep:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if scheduler is not None:
            if config.SCHEDULER_TYPE == 'reduce_on_plateau':
                scheduler.step(vl_acc)
            else:
                scheduler.step()

        if vl_acc > best_val_acc + config.EARLY_STOPPING_MIN_DELTA:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0

            ckpt_path = config.MODELS_DIR / f"best_{config.MODEL_TYPE}_bigru.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config.to_dict(),
            }, ckpt_path)
            print(f"  ✓ New best val acc: {best_val_acc:.2f}%  →  saved to {ckpt_path}")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{config.EARLY_STOPPING_PATIENCE})")

        if config.EARLY_STOPPING and patience_count >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠  Early stopping after epoch {epoch+1}")
            break

        # Log to CSV
        csv_path = config.LOGS_DIR / f"{config.MODEL_TYPE}_training_log.csv"
        log_epoch_to_csv(csv_path, epoch+1, tr_loss, tr_acc, vl_loss, vl_acc)

    total_time = time.time() - t0
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE  —  {total_time/60:.2f} min  |  Best val acc: {best_val_acc:.2f}%")
    print("=" * 80)

    if best_state is not None:
        model.load_state_dict(best_state)

    # FINAL TEST SET EVALUATION
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION (HELD-OUT DATA)")
    print("="*80)
    print("Evaluating best model on test set (never seen during training/validation)")
    
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    
    print(f"\n🎯 TEST SET RESULTS:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"\nComparison:")
    print(f"  Best Val Acc:  {best_val_acc:.2f}%")
    print(f"  Test Acc:      {test_acc:.2f}%")
    print(f"  Gap:           {best_val_acc - test_acc:+.2f}%")
    print("="*80)

    # Save history
    hist_path = config.LOGS_DIR / f"{config.MODEL_TYPE}_training_history.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save results summary
    res_path = config.LOGS_DIR / f"{config.MODEL_TYPE}_results.json"
    with open(res_path, 'w') as f:
        json.dump({
            'model_type': config.MODEL_TYPE,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,  # ← ADD TEST ACCURACY
            'total_epochs': len(history['train_acc']),
            'total_time_minutes': total_time / 60,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc':   history['val_acc'][-1],
            'config': config.to_dict(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    return model, history, best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train BiGRU model for stock signal classification')
    parser.add_argument('--model', type=str, default='hybrid',
                        choices=['deep', 'attention', 'residual', 'hybrid', 'pyramidal'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seq-len', type=int, default=10)
    args = parser.parse_args()

    cfg = get_config(args.model)
    cfg.SEQUENCE_LENGTH = args.seq_len
    if args.epochs:    cfg.NUM_EPOCHS   = args.epochs
    if args.batch_size: cfg.BATCH_SIZE  = args.batch_size
    if args.lr:        cfg.LEARNING_RATE = args.lr

    train_model(cfg)


if __name__ == '__main__':
    main()
