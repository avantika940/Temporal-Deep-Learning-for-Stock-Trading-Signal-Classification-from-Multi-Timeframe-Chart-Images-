"""
Training script for GRU models with fixed dataset (no data leakage)

This script uses the corrected dataset and configuration to eliminate
data leakage and provide realistic results for GRU models.
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime
import json
import csv

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


def train_fixed_gru_model(model_type='deep'):
    """Train GRU model with fixed dataset (no leakage)"""
    
    print("=" * 80)
    print(f"TRAINING GRU {model_type.upper()} MODEL - FIXED VERSION (NO DATA LEAKAGE)")
    print("=" * 80)
    
    # Configuration
    config = FixedConfig(MODEL_TYPE=model_type)
    config.print_config()
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate paths
    config.validate_paths()
    
    # Create data loaders (fixed version)
    try:
        train_loader, val_loader, test_loader = create_fixed_data_loaders(config)
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Please check that the data directory exists and contains the required structure.")
        return None
    
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
                }, config.MODELS_DIR / f'best_fixed_{config.MODEL_TYPE}_gru.pth')
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
        print("   - Verify stratified splits are working correctly")
        print("   - Check sequence diversity statistics")
        print("   - Compare with baseline random classifier (33.3%)")
    
    # Test evaluation
    print(f"\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    config.validate_results(test_acc)
    
    # Save results
    results = {
        'model_type': model_type,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time_minutes': training_time / 60,
        'total_epochs': epoch + 1,
        'history': history,
        'config': config.to_dict(),
        'is_realistic': is_realistic,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = config.LOGS_DIR / f'fixed_{config.MODEL_TYPE}_gru_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Save detailed training history as CSV (like BiLSTM)
    csv_path = config.LOGS_DIR / f'fixed_{config.MODEL_TYPE}_gru_training_curve.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['train_acc'][i],
                history['val_loss'][i],
                history['val_acc'][i],
                history['learning_rates'][i]
            ])
    
    print(f"Training curves saved to: {csv_path}")
    
    return results


def train_all_fixed_gru_models():
    """Train all GRU model types with fixed dataset"""
    
    model_types = ['deep', 'attention', 'residual', 'hybrid', 'pyramidal']
    all_results = {}
    
    # Get base config for paths
    base_config = FixedConfig()
    
    print("=" * 80)
    print("TRAINING ALL GRU MODELS - FIXED VERSION (NO DATA LEAKAGE)")
    print("=" * 80)
    
    for i, model_type in enumerate(model_types):
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL {i+1}/5: {model_type.upper()}")
        print(f"{'='*60}")
        
        try:
            results = train_fixed_gru_model(model_type)
            if results:
                all_results[model_type] = results
                print(f"✅ {model_type} model completed - Val Acc: {results['best_val_acc']:.2f}%")
            else:
                print(f"❌ {model_type} model failed")
        except Exception as e:
            print(f"❌ Error training {model_type} model: {e}")
    
    # Save combined results
    if all_results:
        combined_results_path = base_config.LOGS_DIR / "all_models_comparison.json"
        with open(combined_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("FIXED GRU MODELS - FINAL SUMMARY")
        print("=" * 80)
        
        print("Model       | Val Acc | Test Acc | Time (min) | Realistic")
        print("------------|---------|----------|------------|----------")
        
        for model_type, results in all_results.items():
            val_acc = results['best_val_acc']
            test_acc = results['test_acc']
            time_min = results['training_time_minutes']
            realistic = "✅" if results['is_realistic'] else "⚠️"
            
            print(f"{model_type:11} | {val_acc:6.2f}% | {test_acc:7.2f}% | {time_min:9.1f} | {realistic:8}")
        
        # Save comprehensive experiment summary (like BiLSTM)
        summary_path = base_config.RESULTS_DIR / "EXPERIMENT_SUMMARY.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GRU EXPERIMENTS - COMPREHENSIVE SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}\n")
            
            total_time = sum(r['training_time_minutes'] for r in all_results.values())
            f.write(f"Total Training Time: {total_time:.1f} minutes ({total_time/60:.1f} hours)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("OVERALL RESULTS - ALL 5 GRU MODELS\n")
            f.write("=" * 80 + "\n\n")
            
            # Sort results by validation accuracy
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True)
            
            f.write("Rank  Model       Val Acc    Test Acc   Time (min)  Epochs  Realistic\n")
            f.write("----  ----------  ---------  ---------  ----------  ------  ---------\n")
            
            for i, (model_type, results) in enumerate(sorted_results):
                rank = i + 1
                val_acc = results['best_val_acc']
                test_acc = results['test_acc']
                time_min = results['training_time_minutes']
                epochs = results['total_epochs']
                realistic = "✅" if results['is_realistic'] else "⚠️"
                
                f.write(f" {rank:2d}   {model_type:10}  {val_acc:7.2f}%   {test_acc:7.2f}%   {time_min:8.1f}    {epochs:4d}    {realistic}\n")
            
            f.write(f"\nAverage Training Time per Model: {total_time/len(all_results):.2f} minutes\n")
            f.write(f"Total Experiments Runtime: {total_time:.2f} minutes ({total_time/60:.2f} hours)\n\n")
            
            # Best model details
            best_model, best_results = sorted_results[0]
            f.write("=" * 80 + "\n")
            f.write(f"BEST MODEL: {best_model.upper()} GRU\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Validation Accuracy: {best_results['best_val_acc']:.2f}%\n")
            f.write(f"Test Accuracy: {best_results['test_acc']:.2f}%\n")
            f.write(f"Training Time: {best_results['training_time_minutes']:.2f} minutes\n")
            f.write(f"Epochs: {best_results['total_epochs']}\n")
            f.write(f"Realistic Results: {'Yes' if best_results['is_realistic'] else 'No'}\n\n")
            
            # Data leakage analysis
            f.write("=" * 80 + "\n")
            f.write("DATA LEAKAGE PREVENTION\n")
            f.write("=" * 80 + "\n\n")
            f.write("Fixed Implementation Features:\n")
            f.write("- Sequence Length: 10 images (mentor requirement)\n")
            f.write("- Stratified Split: 60%/20%/20% per class\n")
            f.write("- Aggressive Filtering: 85% homogeneous sequences removed\n")
            f.write("- Stride: 33% overlap (stride=3)\n")
            f.write("- Expected Accuracy: 55-65% (realistic for financial data)\n")
            f.write("- Suspicious Threshold: >70% accuracy\n\n")
            
            realistic_count = sum(1 for r in all_results.values() if r['is_realistic'])
            f.write(f"Results Validation: {realistic_count}/{len(all_results)} models show realistic results\n")
            
            if realistic_count == len(all_results):
                f.write("✅ All models passed realism check - No data leakage detected\n")
            else:
                f.write("⚠️  Some models show suspicious results - Further investigation needed\n")
        
        print(f"Experiment summary saved to: {summary_path}")
        
        print(f"\nCombined results saved to: {combined_results_path}")
        
        # Check if any results are still unrealistic
        unrealistic_count = sum(1 for r in all_results.values() if not r['is_realistic'])
        if unrealistic_count > 0:
            print(f"\n⚠️  WARNING: {unrealistic_count}/{len(all_results)} models still show unrealistic results!")
            print("   Further investigation may be needed.")
        else:
            print(f"\n✅ All {len(all_results)} models show realistic results!")


if __name__ == "__main__":
    train_all_fixed_gru_models()
