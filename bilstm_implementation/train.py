"""
Training script for BiLSTM models

This script handles the complete training pipeline including:
- Model initialization
- Training loop with progress tracking
- Validation
- Early stopping
- Model checkpointing
- Logging
"""

import os
import sys
from pathlib import Path
import argparse
import json
import csv
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import Config, get_config
from dataset import create_data_loaders
from models import create_model


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
        # Update progress bar
        current_acc = 100 * correct / total
        current_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for sequences, labels in pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            # Update progress bar
            current_acc = 100 * correct / total
            current_loss = running_loss / (len(pbar.iterable) if hasattr(pbar, 'iterable') else 1)
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


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


def train_model(config):
    """
    Main training function
    
    Args:
        config: Configuration object
    """
    # Print configuration
    config.print_config()
    
    # Validate paths
    config.validate_paths()
    
    # Set seed
    set_seed(config.RANDOM_SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.DEVICE = str(device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    model = create_model(config)
    model = model.to(device)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    # Optimizer
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS
        )
    elif config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS
        )
    elif config.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    # Learning rate scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE,
                min_lr=config.SCHEDULER_MIN_LR
            )
        elif config.SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.T_MAX,
                eta_min=config.ETA_MIN
            )
        elif config.SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.STEP_SIZE,
                gamma=config.GAMMA
            )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Training state
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    start_time = time.time()
    
    # Initialize CSV logging
    csv_path = config.LOGS_DIR / f"training_curve_{config.MODEL_TYPE}_bilstm.csv"
    init_csv_logging(csv_path)
    
    # Training loop
    print("\n" + "="*80)
    print(f"TRAINING {config.MODEL_TYPE.upper()} MODEL")
    print("="*80)
    print(f"Training for {config.NUM_EPOCHS} epochs")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Optimizer: {config.OPTIMIZER}")
    if scheduler:
        print(f"Scheduler: {config.SCHEDULER_TYPE}")
    print("="*80 + "\n")
    
    # Initialize CSV logging
    csv_path = config.LOGS_DIR / f"training_curve_{config.MODEL_TYPE}_bilstm.csv"
    init_csv_logging(csv_path)
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Log to CSV
        log_epoch_to_csv(csv_path, epoch+1, train_loss, train_acc, val_loss, val_acc)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if config.SCHEDULER_TYPE == 'reduce_on_plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Check for improvement
        improved = False
        if val_acc > best_val_acc + config.EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improved = True
            
            # Save best model
            model_path = config.MODELS_DIR / f"best_{config.MODEL_TYPE}_bilstm.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config.to_dict()
            }, model_path)
            
            print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
            print(f"  ✓ Model saved to: {model_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
        # Early stopping
        if config.EARLY_STOPPING and patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.MODELS_DIR / f'best_{config.MODEL_TYPE}_bilstm.pth'}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # FINAL TEST SET EVALUATION (on truly held-out data)
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
    
    # Save training history
    history_path = config.LOGS_DIR / f"{config.MODEL_TYPE}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save final results summary
    results = {
        'model_type': config.MODEL_TYPE,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,  # ← ADD TEST ACCURACY
        'total_epochs': len(history['train_acc']),
        'total_time_minutes': total_time / 60,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'config': config.to_dict(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_path = config.LOGS_DIR / f"{config.MODEL_TYPE}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    print("="*80 + "\n")
    
    return model, history, best_val_acc


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train BiLSTM model for stock signal classification')
    
    parser.add_argument('--model', type=str, default='hybrid',
                       choices=['deep', 'attention', 'residual', 'hybrid', 'pyramidal'],
                       help='Model type to train')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Sequence length (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Number of LSTM layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create config
    config = get_config(args.model)
    
    # Override with command line arguments
    config.SEQUENCE_LENGTH = args.sequence_length
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.HIDDEN_DIM = args.hidden_dim
    config.NUM_LSTM_LAYERS = args.num_layers
    config.LSTM_DROPOUT = args.dropout
    config.EARLY_STOPPING = not args.no_early_stopping
    config.RANDOM_SEED = args.seed
    
    # Train model
    try:
        model, history, best_acc = train_model(config)
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Best validation accuracy: {best_acc:.2f}%")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
