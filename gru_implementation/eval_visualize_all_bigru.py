"""
Evaluate PyramidalBiGRU + generate comparison chart for ALL 5 BiGRU models.

Actions:
  1. Load each model's saved checkpoint and evaluate on the validation set.
  2. Save pyramidal_results.json (missing).
  3. Generate single two-panel comparison chart (accuracy + precision/recall/F1).

Run:
    python eval_visualize_all_bigru.py
"""

import sys, json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dataset import create_data_loaders
from models import create_model

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_TYPES     = ['deep', 'attention', 'residual', 'hybrid', 'pyramidal']
STEP4_BASELINE  = 61.21   # Step-4 Temporal Transformer
BILSTM_BEST     = 75.57   # Hybrid BiLSTM best result
LOGS_DIR        = Config.LOGS_DIR
MODELS_DIR      = Config.MODELS_DIR
VIZ_DIR         = Config.VIZ_DIR
CLASS_NAMES     = Config.CLASS_NAMES

VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _serial(obj):
    if isinstance(obj, (np.integer, np.floating)): return float(obj)
    if isinstance(obj, np.ndarray):               return obj.tolist()
    raise TypeError


def evaluate_model(model, val_loader, device) -> dict:
    model.eval()
    preds, targets, probs_list = [], [], []

    with torch.no_grad():
        for seqs, labels in tqdm(val_loader, desc='  evaluating', leave=False):
            seqs  = seqs.to(device)
            out   = model(seqs)
            prob  = torch.softmax(out, dim=1)
            _, p  = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            targets.extend(labels.numpy())
            probs_list.extend(prob.cpu().numpy())

    preds   = np.array(preds)
    targets = np.array(targets)

    acc  = accuracy_score(targets, preds) * 100
    prec = precision_score(targets, preds, average='weighted', zero_division=0) * 100
    rec  = recall_score(targets,   preds, average='weighted', zero_division=0) * 100
    f1   = f1_score(targets,       preds, average='weighted', zero_division=0) * 100

    return {
        'accuracy':            float(acc),
        'precision':           float(prec),
        'recall':              float(rec),
        'f1_score':            float(f1),
        'precision_per_class': (precision_score(targets, preds, average=None, zero_division=0) * 100).tolist(),
        'recall_per_class':    (recall_score(targets,   preds, average=None, zero_division=0) * 100).tolist(),
        'f1_per_class':        (f1_score(targets,       preds, average=None, zero_division=0) * 100).tolist(),
        'confusion_matrix':    confusion_matrix(targets, preds).tolist(),
        'classification_report': classification_report(targets, preds,
                                     target_names=CLASS_NAMES, digits=4),
    }


# ── Comparison chart ──────────────────────────────────────────────────────────

def plot_comparison(all_results: dict):
    """Two-panel comparison chart matching the BiLSTM reference style."""
    models_asc  = sorted(all_results, key=lambda m: all_results[m]['accuracy'])
    models_desc = list(reversed(models_asc))
    labels_asc  = [m.capitalize() for m in models_asc]
    labels_desc = [m.capitalize() for m in models_desc]

    accs  = [all_results[m]['accuracy']  for m in models_asc]
    precs = [all_results[m]['precision'] for m in models_desc]
    recs  = [all_results[m]['recall']    for m in models_desc]
    f1s   = [all_results[m]['f1_score']  for m in models_desc]

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle(
        'BiGRU Models — Cross-Class Timeline  |  seq_len=10\n'
        'Adani Stock Trading Signal Classification (BUY / HOLD / SELL)',
        fontsize=14, fontweight='bold', y=1.01,
    )

    bar_kw = dict(edgecolor='#333', linewidth=0.8)
    BLUE   = '#4472C4'

    # ── Panel 1: Accuracy (horizontal bars, ascending) ─────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(labels_asc, accs, color=BLUE, **bar_kw)
    ax1.axvline(STEP4_BASELINE, color='red',      linestyle='--', linewidth=1.5,
                label=f'Step 4 Baseline ({STEP4_BASELINE}%)')
    ax1.axvline(BILSTM_BEST,    color='darkgreen', linestyle=':',  linewidth=1.5,
                label=f'BiLSTM Hybrid ({BILSTM_BEST}%)')
    for bar, v in zip(bars, accs):
        ax1.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{v:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xlim(0, max(accs) * 1.15)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis='x', alpha=0.3)

    # ── Panel 2: Precision / Recall / F1 (grouped bars, descending acc) ───
    ax2 = fig.add_subplot(gs[0, 1])
    xi  = np.arange(len(labels_desc)); w = 0.25
    ax2.bar(xi - w, precs, width=w, label='Precision', color='#4472C4', **bar_kw)
    ax2.bar(xi,     recs,  width=w, label='Recall',    color='#ED7D31', **bar_kw)
    ax2.bar(xi + w, f1s,   width=w, label='F1-Score',  color='#70AD47', **bar_kw)
    ax2.set_xticks(xi); ax2.set_xticklabels(labels_desc, fontsize=9)
    ax2.set_ylabel('Score (%)')
    ax2.set_title('Precision, Recall, F1-Score Comparison')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = VIZ_DIR / 'bigru_model_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison chart → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("  EVALUATE ALL BiGRU MODELS + GENERATE COMPARISON CHART")
    print(f"  Device: {device}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    config = Config(MODEL_TYPE='deep', SEQUENCE_LENGTH=10)
    _, val_loader = create_data_loaders(config)

    all_results = {}

    for mt in MODEL_TYPES:
        print(f"\n{'─'*60}")
        print(f"  [{MODEL_TYPES.index(mt)+1}/5]  {mt.upper()} BiGRU")
        print(f"{'─'*60}")

        ckpt_path = MODELS_DIR / f'best_{mt}_bigru.pth'
        if not ckpt_path.exists():
            print(f"  ✗  Checkpoint not found: {ckpt_path} — skipping")
            continue

        cfg   = Config(MODEL_TYPE=mt, SEQUENCE_LENGTH=10)
        model = create_model(cfg).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_epoch = ckpt.get('epoch', '?')
        saved_vacc = ckpt.get('val_acc', 0.0)
        print(f"  Loaded checkpoint (best epoch={best_epoch}, saved val_acc={saved_vacc:.2f}%)")

        metrics = evaluate_model(model, val_loader, device)
        all_results[mt] = metrics
        print(f"  Val Acc={metrics['accuracy']:.2f}%  "
              f"Prec={metrics['precision']:.2f}%  "
              f"Rec={metrics['recall']:.2f}%  "
              f"F1={metrics['f1_score']:.2f}%")

        # Save results JSON if missing
        res_path = LOGS_DIR / f'{mt}_results.json'
        if not res_path.exists():
            hist_path = LOGS_DIR / f'{mt}_training_history.json'
            history   = {}
            if hist_path.exists():
                with open(hist_path) as fh:
                    history = json.load(fh)
            res_dict = {
                'model_type':         mt,
                'best_val_acc':       metrics['accuracy'],
                'total_epochs':       best_epoch,
                'total_time_minutes': None,
                'final_train_acc':    history.get('train_acc', [None])[-1],
                'final_val_acc':      history.get('val_acc',   [metrics['accuracy']])[-1],
                'config':             {'MODEL_TYPE': mt, 'SEQUENCE_LENGTH': 10},
                'timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(res_path, 'w') as fh:
                json.dump(res_dict, fh, indent=2, default=_serial)
            print(f"  Saved results  → {res_path}")

    # Summary leaderboard
    print(f"\n{'='*70}")
    print("  FINAL LEADERBOARD — BiGRU | seq_len=10 | Cross-Class")
    print(f"{'='*70}")
    print(f"  {'Model':<12} {'Val Acc':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "─" * 52)
    for mt in sorted(all_results, key=lambda m: all_results[m]['accuracy'], reverse=True):
        r = all_results[mt]
        print(f"  {mt.capitalize():<12} {r['accuracy']:>8.2f}%  "
              f"{r['precision']:>8.2f}%  {r['recall']:>6.2f}%  {r['f1_score']:>6.2f}%")
    print(f"\n  Step-4 Baseline:  {STEP4_BASELINE:.2f}%")
    print(f"  BiLSTM Hybrid:    {BILSTM_BEST:.2f}%")
    best_gru = max(all_results, key=lambda m: all_results[m]['accuracy'])
    print(f"  BiGRU Best ({best_gru.capitalize()}): {all_results[best_gru]['accuracy']:.2f}%")
    print(f"{'='*70}")

    plot_comparison(all_results)
    print(f"\n  Done — chart saved to: {VIZ_DIR}")
    print()


if __name__ == '__main__':
    main()
