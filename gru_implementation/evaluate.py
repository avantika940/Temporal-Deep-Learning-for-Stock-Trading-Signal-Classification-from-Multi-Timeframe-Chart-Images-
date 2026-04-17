"""
Evaluation script for trained BiGRU models.

Provides:
  - Accuracy, Precision, Recall, F1-Score (weighted)
  - Per-class breakdown
  - Confusion matrix visualisation
  - Training curves plot
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from dataset import create_data_loaders
from models import create_model


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, data_loader, device, config) -> dict:
    """
    Evaluate a trained BiGRU model.

    Returns a dictionary with all scalar metrics plus raw predictions.
    """
    model.eval()
    preds, targets, probs_list = [], [], []

    print("\nEvaluating model...")
    with torch.no_grad():
        for seqs, labels in tqdm(data_loader, desc='Evaluating'):
            seqs = seqs.to(device)
            out  = model(seqs)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.numpy())
            probs_list.extend(prob.cpu().numpy())

    preds   = np.array(preds)
    targets = np.array(targets)
    probs   = np.array(probs_list)

    acc  = accuracy_score(targets, preds) * 100
    prec = precision_score(targets, preds, average='weighted', zero_division=0) * 100
    rec  = recall_score(targets,  preds,  average='weighted', zero_division=0) * 100
    f1   = f1_score(targets,     preds,  average='weighted', zero_division=0) * 100

    prec_cls = precision_score(targets, preds, average=None, zero_division=0) * 100
    rec_cls  = recall_score(targets,  preds,  average=None, zero_division=0) * 100
    f1_cls   = f1_score(targets,     preds,  average=None, zero_division=0) * 100
    cm       = confusion_matrix(targets, preds)
    report   = classification_report(targets, preds,
                                     target_names=config.CLASS_NAMES, digits=4)

    return {
        'accuracy':            float(acc),
        'precision':           float(prec),
        'recall':              float(rec),
        'f1_score':            float(f1),
        'precision_per_class': prec_cls.tolist(),
        'recall_per_class':    rec_cls.tolist(),
        'f1_per_class':        f1_cls.tolist(),
        'confusion_matrix':    cm.tolist(),
        'classification_report': report,
        'predictions':         preds.tolist(),
        'labels':              targets.tolist(),
        'probabilities':       probs.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, save_path):
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Confusion Matrix (Normalised)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label');  plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix → {save_path}")


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'],   'r-', label='Val',   linewidth=2)
    axes[0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'],   'r-', label='Val',   linewidth=2)
    best_ep  = int(np.argmax(history['val_acc'])) + 1
    best_acc = max(history['val_acc'])
    axes[1].axvline(best_ep, color='g', linestyle='--', alpha=0.5)
    axes[1].text(best_ep, best_acc, f'  Best: {best_acc:.2f}%',
                 va='center', fontsize=9, color='green')
    axes[1].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves → {save_path}")


def plot_per_class_performance(metrics, class_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(class_names)); w = 0.25
    ax.bar(x - w, metrics['precision_per_class'], w, label='Precision', color='steelblue')
    ax.bar(x,     metrics['recall_per_class'],    w, label='Recall',    color='coral')
    ax.bar(x + w, metrics['f1_per_class'],        w, label='F1-Score',  color='lightgreen')
    for i, (p, r, f) in enumerate(zip(metrics['precision_per_class'],
                                       metrics['recall_per_class'],
                                       metrics['f1_per_class'])):
        ax.text(i - w, p + 0.5, f'{p:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i,     r + 0.5, f'{r:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + w, f + 0.5, f'{f:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylabel('Score (%)'); ax.set_ylim(0, 110)
    ax.set_title('Per-Class Performance', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class plot → {save_path}")


def print_evaluation_results(metrics, config):
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']:.2f}%")
    print(f"\nPer-Class Performance:")
    for i, cls in enumerate(config.CLASS_NAMES):
        print(f"  {cls}:")
        print(f"    Precision: {metrics['precision_per_class'][i]:.2f}%")
        print(f"    Recall:    {metrics['recall_per_class'][i]:.2f}%")
        print(f"    F1-Score:  {metrics['f1_per_class'][i]:.2f}%")
    print(f"\nDetailed Classification Report:")
    print(metrics['classification_report'])
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained BiGRU model')
    parser.add_argument('--model', type=str, default='hybrid',
                        choices=['deep', 'attention', 'residual', 'hybrid', 'pyramidal'])
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--seq-len', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    config = get_config(args.model)
    config.SEQUENCE_LENGTH = args.seq_len
    config.EVAL_BATCH_SIZE = args.batch_size
    config.print_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    _, val_loader, test_loader = create_data_loaders(config)

    model = create_model(config).to(device)

    ckpt_path = Path(args.model_path) if args.model_path \
        else config.MODELS_DIR / f"best_{config.MODEL_TYPE}_bigru.pth"

    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        print("Please train the model first using train.py")
        return 1

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded  (epoch {ckpt['epoch']}, best val acc {ckpt['val_acc']:.2f}%)")

    # Evaluate on validation set
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    metrics = evaluate_model(model, val_loader, device, config)
    print("\n📊 VALIDATION SET RESULTS:")
    print_evaluation_results(metrics, config)

    # Evaluate on test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION (HELD-OUT DATA)")
    print("="*80)
    test_metrics = evaluate_model(model, test_loader, device, config)
    print("\n🎯 TEST SET RESULTS:")
    print_evaluation_results(test_metrics, config)
    
    # Comparison
    print("\n📊 COMPARISON (Val vs Test):")
    print("="*80)
    print(f"  Validation Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Test Accuracy:       {test_metrics['accuracy']:.2f}%")
    print(f"  Gap:                 {metrics['accuracy'] - test_metrics['accuracy']:+.2f}%")
    print("="*80)

    # Save full metrics JSON
    out = config.LOGS_DIR / f"{config.MODEL_TYPE}_evaluation_metrics.json"
    with open(out, 'w') as f:
        save_metrics = {
            'validation': {k: v for k, v in metrics.items()
                          if k not in ('predictions', 'labels', 'probabilities')},
            'test': {k: v for k, v in test_metrics.items()
                    if k not in ('predictions', 'labels', 'probabilities')}
        }
        json.dump(save_metrics, f, indent=2)
    print(f"\nMetrics saved → {out}")

    # Optional plots (skip if history file is missing)
    hist_path = config.LOGS_DIR / f"{config.MODEL_TYPE}_training_history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_curves(history, config.VIZ_DIR / f"{config.MODEL_TYPE}_training_curves.png")

    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        config.CLASS_NAMES,
        config.VIZ_DIR / f"{config.MODEL_TYPE}_confusion_matrix.png",
    )
    plot_per_class_performance(
        metrics,
        config.CLASS_NAMES,
        config.VIZ_DIR / f"{config.MODEL_TYPE}_per_class_performance.png",
    )

    return 0


if __name__ == '__main__':
    main()
