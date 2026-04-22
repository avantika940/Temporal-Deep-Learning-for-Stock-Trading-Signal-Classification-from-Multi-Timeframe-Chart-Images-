"""
Run all 5 BiLSTM model variants with fixed dataset (no data leakage)

This script trains all model architectures:
1. Deep BiLSTM
2. Attention BiLSTM  
3. Residual BiLSTM
4. Hybrid BiLSTM
5. Pyramidal BiLSTM

Each model is trained with proper temporal splits and realistic expectations.
"""

import os
import sys
from pathlib import Path
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config_fixed import FixedConfig
from dataset_fixed import create_fixed_data_loaders
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


def train_single_model(model_type, train_loader, val_loader, test_loader, base_config, device):
    """Train a single model variant"""
    
    print(f"\n" + "="*80)
    print(f"TRAINING {model_type.upper()} MODEL - FIXED VERSION")
    print(f"="*80)
    
    # Create config for this model type
    config = FixedConfig()
    config.MODEL_TYPE = model_type
    
    # Adjust architecture based on model type
    if model_type == 'pyramidal':
        config.HIDDEN_DIM = 64  # Smaller for pyramidal
        config.NUM_LSTM_LAYERS = 4  # More layers
    elif model_type == 'deep':
        config.NUM_LSTM_LAYERS = 3  # More layers for deep
        config.LSTM_DROPOUT = 0.4  # Slightly less dropout
    elif model_type == 'attention':
        config.ATTENTION_HEADS = 6  # More attention heads
        config.NUM_LSTM_LAYERS = 2  # Fewer LSTM layers
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Create model
    try:
        model = create_model(config).to(device)
        print(f"Model created: {model_type}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"Error creating {model_type} model: {e}")
        return None
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config.to_dict()
            }, config.MODELS_DIR / f'best_fixed_{model_type}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Test evaluation
    print(f"\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Validate results
    is_realistic = config.validate_results(best_val_acc)
    
    if not is_realistic:
        print("⚠️  Results may still have data leakage issues!")
    else:
        print("✅ Results appear realistic for financial time series.")
    
    # Calculate additional metrics
    precision = recall = f1_score = 0.0
    
    # Collect results
    results = {
        'model_type': model_type,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'training_time_minutes': training_time / 60,
        'total_epochs': epoch + 1,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'history': history,
        'is_realistic': is_realistic,
        'config': {
            'hidden_dim': config.HIDDEN_DIM,
            'num_layers': config.NUM_LSTM_LAYERS,
            'sequence_length': config.SEQUENCE_LENGTH,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE
        }
    }
    
    # Save individual results
    results_path = config.LOGS_DIR / f'fixed_{model_type}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results


def run_all_models():
    """Run all 5 BiLSTM model variants with fixed dataset"""
    
    print("=" * 80)
    print("BILSTM ALL MODELS EXPERIMENT - FIXED VERSION (NO DATA LEAKAGE)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Base configuration
    base_config = FixedConfig()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders once (they're the same for all models)
    print("\nCreating fixed data loaders...")
    try:
        train_loader, val_loader, test_loader = create_fixed_data_loaders(base_config)
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Model types to train
    model_types = ['deep', 'attention', 'residual', 'hybrid', 'pyramidal']
    
    # Results storage
    all_results = {}
    experiment_start_time = time.time()
    
    # Train each model
    for i, model_type in enumerate(model_types, 1):
        print(f"\n{'='*20} MODEL {i}/5: {model_type.upper()} {'='*20}")
        
        try:
            results = train_single_model(
                model_type, train_loader, val_loader, test_loader, base_config, device
            )
            
            if results:
                all_results[model_type] = results
                print(f"✅ {model_type} completed successfully")
            else:
                print(f"❌ {model_type} failed")
                
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            continue
    
    # Experiment completed
    total_time = time.time() - experiment_start_time
    
    # Generate summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY - FIXED RESULTS")
    print(f"{'='*80}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/60:.2f} minutes")
    
    if all_results:
        print(f"\nResults Summary:")
        print("Rank  Model       Val Acc   Test Acc  Time(min)  Realistic")
        print("-" * 60)
        
        # Sort by validation accuracy
        sorted_results = sorted(all_results.items(), 
                               key=lambda x: x[1]['best_val_acc'], 
                               reverse=True)
        
        for i, (model_type, results) in enumerate(sorted_results, 1):
            realistic_icon = "✅" if results['is_realistic'] else "⚠️"
            print(f" {i:2d}   {model_type:10s}  {results['best_val_acc']:6.2f}%   "
                  f"{results['test_acc']:7.2f}%  {results['training_time_minutes']:8.1f}   {realistic_icon}")
        
        # Check for realistic results
        realistic_count = sum(1 for r in all_results.values() if r['is_realistic'])
        print(f"\nRealistic results: {realistic_count}/{len(all_results)}")
        
        if realistic_count == len(all_results):
            print("✅ All results appear realistic for financial time series!")
        elif realistic_count > 0:
            print("⚠️  Some results may still have data leakage issues.")
        else:
            print("❌ All results are unrealistic - check for remaining data leakage!")
    
    # Save combined results
    combined_results = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_minutes': total_time / 60,
        'results': all_results,
        'summary': {
            'best_model': max(all_results.items(), key=lambda x: x[1]['best_val_acc'])[0] if all_results else None,
            'avg_val_acc': np.mean([r['best_val_acc'] for r in all_results.values()]) if all_results else 0,
            'avg_test_acc': np.mean([r['test_acc'] for r in all_results.values()]) if all_results else 0,
            'realistic_count': sum(1 for r in all_results.values() if r['is_realistic'])
        }
    }
    
    # Save to file
    combined_path = base_config.LOGS_DIR / 'all_models_fixed_results.json'
    with open(combined_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined results saved to: {combined_path}")
    
    return combined_results


if __name__ == "__main__":
    results = run_all_models()
