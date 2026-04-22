"""
Training script using fixed dataset with no data leakage

This script uses the corrected dataset and configuration to eliminate
data leakage and provide realistic results.
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import FixedConfig
from dataset import create_fixed_data_loaders
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
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
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
    
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc='Validating', leave=False):
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_fixed_model():
    """Train model with fixed dataset (no leakage)"""
    
    print("=" * 80)
    print("TRAINING BILSTM MODEL - FIXED VERSION (NO DATA LEAKAGE)")
    print("=" * 80)
    
    # Configuration
    config = FixedConfig()
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders (fixed version)
    try:
        train_loader, val_loader, test_loader = create_fixed_data_loaders(config)
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Please check that the data directory exists and contains the required structure.")
        return
    
    # Create model
    model = create_model(config).to(device)
    print(f"Model created: {config.MODEL_TYPE}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=config.BETAS
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR
    )
    
    # Training state
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print(f"Expected accuracy range: {config.EXPECTED_ACCURACY_RANGE[0]}-{config.EXPECTED_ACCURACY_RANGE[1]}%")
    print(f"Suspicious if > {config.SUSPICIOUS_ACCURACY_THRESHOLD}%")
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Check for suspicious accuracy
        if val_acc > config.SUSPICIOUS_ACCURACY_THRESHOLD:
            print(f"⚠️  WARNING: Validation accuracy {val_acc:.2f}% is suspiciously high!")
            print("   This may indicate remaining data leakage issues.")
        
        # Early stopping
        if val_acc > best_val_acc + config.EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            if config.SAVE_BEST_ONLY:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config.to_dict()
                }, config.MODELS_DIR / f'best_fixed_{config.MODEL_TYPE}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Validate results
    print("\n" + "=" * 80)
    print("RESULT VALIDATION")
    print("=" * 80)
    
    is_realistic = config.validate_results(best_val_acc)
    
    if not is_realistic:
        print("\n⚠️  RECOMMENDATION: Check for remaining data leakage issues!")
        print("   - Verify temporal splits have no overlap")
        print("   - Check sequence diversity statistics")
        print("   - Compare with baseline random classifier (33.3%)")
    
    # Test evaluation
    print(f"\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    config.validate_results(test_acc)
    
    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time_minutes': training_time / 60,
        'total_epochs': epoch + 1,
        'history': history,
        'config': config.to_dict(),
        'is_realistic': is_realistic
    }
    
    import json
    results_path = config.LOGS_DIR / f'fixed_{config.MODEL_TYPE}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    results = train_fixed_model()
