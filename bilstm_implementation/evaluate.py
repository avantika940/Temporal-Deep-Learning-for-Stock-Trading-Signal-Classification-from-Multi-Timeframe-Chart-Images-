"""
Evaluation script for trained BiLSTM models

Provides comprehensive evaluation including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Per-class performance analysis
- Training curves plotting
"""

import os
import sys
from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import Config, get_config
from dataset import create_data_loaders
from models import create_model


def evaluate_model(model, data_loader, device, config):
    """
    Evaluate model on given data loader
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to use
        config: Configuration
        
    Returns:
        Dictionary containing all metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for sequences, labels in tqdm(data_loader, desc='Evaluating'):
            sequences = sequences.to(device)
            
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, average='weighted') * 100
    recall = recall_score(all_labels, all_predictions, average='weighted') * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_predictions, average=None) * 100
    recall_per_class = recall_score(all_labels, all_predictions, average=None) * 100
    f1_per_class = f1_score(all_labels, all_predictions, average=None) * 100
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(
        all_labels, all_predictions,
        target_names=config.CLASS_NAMES,
        digits=4
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist()
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_training_curves(history, save_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Add best validation accuracy marker
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    axes[1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    axes[1].text(best_epoch, best_acc, f'  Best: {best_acc:.2f}%',
                 verticalalignment='center', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")


def plot_per_class_performance(metrics, class_names, save_path):
    """Plot per-class performance metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    precision = metrics['precision_per_class']
    recall = metrics['recall_per_class']
    f1 = metrics['f1_per_class']
    
    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='coral')
    ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 1, f'{p:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, r + 1, f'{r:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, f + 1, f'{f:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class performance plot saved to: {save_path}")


def print_evaluation_results(metrics, config):
    """Print detailed evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']:.2f}%")
    
    print(f"\nPer-Class Performance:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision_per_class'][i]:.2f}%")
        print(f"    Recall:    {metrics['recall_per_class'][i]:.2f}%")
        print(f"    F1-Score:  {metrics['f1_per_class'][i]:.2f}%")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print("         " + "  ".join([f"{cls:>8}" for cls in config.CLASS_NAMES]))
    for i, row in enumerate(cm):
        print(f"{config.CLASS_NAMES[i]:>8} " + "  ".join([f"{val:>8}" for val in row]))
    
    print(f"\nDetailed Classification Report:")
    print(metrics['classification_report'])
    
    print("="*80 + "\n")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained BiLSTM model')
    
    parser.add_argument('--model', type=str, default='hybrid',
                       choices=['deep', 'attention', 'residual', 'hybrid', 'pyramidal'],
                       help='Model type to evaluate')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model checkpoint (optional)')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='Sequence length (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.model)
    config.SEQUENCE_LENGTH = args.sequence_length
    config.EVAL_BATCH_SIZE = args.batch_size
    
    config.print_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create data loaders
    _, val_loader, test_loader = create_data_loaders(config)
    
    # Create model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model = create_model(config)
    model = model.to(device)
    
    # Load model weights
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = config.MODELS_DIR / f"best_{config.MODEL_TYPE}_bilstm.pth"
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return 1
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    print(f"Best validation accuracy during training: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate model
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    metrics = evaluate_model(model, val_loader, device, config)
    
    # Print validation results
    print("\n📊 VALIDATION SET RESULTS:")
    print("="*80)
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall:    {metrics['recall']:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']:.2f}%")
    
    # Now evaluate on test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION (HELD-OUT DATA)")
    print("="*80)
    print("Evaluating on truly held-out test set...")
    
    test_metrics = evaluate_model(model, test_loader, device, config)
    
    # Print test results
    print("\n🎯 TEST SET RESULTS:")
    print("="*80)
    print(f"Accuracy:  {test_metrics['accuracy']:.2f}%")
    print(f"Precision: {test_metrics['precision']:.2f}%")
    print(f"Recall:    {test_metrics['recall']:.2f}%")
    print(f"F1-Score:  {test_metrics['f1_score']:.2f}%")
    
    print("\n📊 COMPARISON (Val vs Test):")
    print("="*80)
    print(f"  Validation Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Test Accuracy:       {test_metrics['accuracy']:.2f}%")
    print(f"  Gap:                 {metrics['accuracy'] - test_metrics['accuracy']:+.2f}%")
    print("="*80)
    
    # Save both metrics
    metrics_path = config.LOGS_DIR / f"{config.MODEL_TYPE}_evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        # Save both val and test metrics
        metrics_to_save = {
            'validation': {k: v for k, v in metrics.items() 
                          if k not in ['predictions', 'labels', 'probabilities']},
            'test': {k: v for k, v in test_metrics.items() 
                    if k not in ['predictions', 'labels', 'probabilities']}
        }
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Plot confusion matrix
    if config.SAVE_CONFUSION_MATRIX:
        cm_path = config.VIZ_DIR / f"{config.MODEL_TYPE}_confusion_matrix.png"
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            config.CLASS_NAMES,
            cm_path
        )
    
    # Plot per-class performance
    perf_path = config.VIZ_DIR / f"{config.MODEL_TYPE}_per_class_performance.png"
    plot_per_class_performance(metrics, config.CLASS_NAMES, perf_path)
    
    # Plot training curves if history available
    history_path = config.LOGS_DIR / f"{config.MODEL_TYPE}_training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        curves_path = config.VIZ_DIR / f"{config.MODEL_TYPE}_training_curves.png"
        plot_training_curves(history, curves_path)
    
    print("\n✓ Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
