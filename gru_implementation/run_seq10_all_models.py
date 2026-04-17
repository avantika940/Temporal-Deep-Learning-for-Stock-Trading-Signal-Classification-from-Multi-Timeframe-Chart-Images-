"""
Run all 5 BiGRU model variants on the cross-class seq_len=10 timeline dataset
and generate a comparison chart (accuracy + precision/recall/F1).

Baseline reference: Step-4 Temporal Transformer  →  61.21 % val accuracy
BiLSTM best (Hybrid): 75.57 %

Usage:
    python run_seq10_all_models.py
"""

import sys, json, time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from train import train_model, set_seed
from evaluate import evaluate_model
from dataset import create_data_loaders
from models import create_model

# ── Shared configuration ──────────────────────────────────────────────────────
BASE_CONFIG = dict(
    SEQUENCE_LENGTH = 10,
    EARLY_STOPPING  = False,   # run full epochs for fair comparison
)

MODEL_TYPES = ['deep', 'attention', 'residual', 'hybrid', 'pyramidal']

STEP4_BASELINE = 61.21   # Step-4 Temporal Transformer val accuracy
BILSTM_BEST    = 75.57   # Hybrid BiLSTM best result


# ── Train all models ──────────────────────────────────────────────────────────
def run_all() -> dict:
    all_results = {}

    print("\n" + "=" * 80)
    print("SEQ=10 CROSS-CLASS EXPERIMENT  —  ALL 5 BiGRU MODELS")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    for i, mt in enumerate(MODEL_TYPES, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/5]  {mt.upper()} BiGRU")
        print(f"{'='*70}\n")

        config = Config(MODEL_TYPE=mt, **BASE_CONFIG)
        set_seed(config.RANDOM_SEED)

        t0 = time.time()
        model, history, best_val_acc = train_model(config)
        elapsed_min = (time.time() - t0) / 60

        # Evaluate best checkpoint on validation set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, val_loader = create_data_loaders(config)
        ckpt = torch.load(config.MODELS_DIR / f'best_{mt}_bigru.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        metrics = evaluate_model(model, val_loader, device, config)

        all_results[mt] = {
            'accuracy':       metrics['accuracy'],
            'precision':      metrics['precision'],
            'recall':         metrics['recall'],
            'f1_score':       metrics['f1_score'],
            'best_epoch':     ckpt['epoch'],
            'training_time':  elapsed_min,
            'epochs_trained': len(history['train_acc']),
            'history':        history,
        }
        print(f"\n  ✓ {mt.upper()} BiGRU:  Val Acc = {metrics['accuracy']:.2f}%  "
              f"F1 = {metrics['f1_score']:.2f}%  Time = {elapsed_min:.1f} min")

    # Save JSON
    out_json = Config.LOGS_DIR / 'seq10_all_bigru_results.json'

    def _serial(obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray):               return obj.tolist()
        raise TypeError

    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=_serial)
    print(f"\nResults saved → {out_json}")

    return all_results


# ── Comparison chart ──────────────────────────────────────────────────────────
def plot_comparison(results: dict):
    """Generate a 2-panel comparison chart (accuracy + precision/recall/F1)."""
    models_sorted = sorted(results, key=lambda m: results[m]['accuracy'])
    labels = [m.capitalize() for m in models_sorted]
    accs   = [results[m]['accuracy'] for m in models_sorted]

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle(
        'BiGRU Models — Cross-Class Timeline  |  seq_len=10\n'
        'Adani Stock Trading Signal Classification (BUY / HOLD / SELL)',
        fontsize=14, fontweight='bold', y=1.01,
    )

    bar_kw   = dict(edgecolor='#333', linewidth=0.8)
    BLUE     = '#4472C4'
    label_kw = dict(ha='left', va='center', fontsize=9, fontweight='bold')

    # ── Panel 1: Accuracy ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(labels, accs, color=BLUE, **bar_kw)
    ax1.axvline(STEP4_BASELINE, color='red', linestyle='--', linewidth=1.5,
                label=f'Step-4 Baseline ({STEP4_BASELINE}%)')
    ax1.axvline(BILSTM_BEST, color='green', linestyle=':', linewidth=1.5,
                label=f'BiLSTM Hybrid ({BILSTM_BEST}%)')
    for bar, v in zip(bars, accs):
        ax1.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{v:.2f}%', **label_kw)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xlim(0, max(accs) * 1.15)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis='x', alpha=0.3)

    # ── Panel 2: Precision / Recall / F1 ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ms2  = sorted(results, key=lambda m: results[m]['accuracy'], reverse=True)
    lbl2 = [m.capitalize() for m in ms2]
    p2   = [results[m]['precision'] for m in ms2]
    r2   = [results[m]['recall']    for m in ms2]
    f2   = [results[m]['f1_score']  for m in ms2]
    xi   = np.arange(len(lbl2)); w = 0.25
    ax2.bar(xi - w, p2, width=w, label='Precision', color='#4472C4', **bar_kw)
    ax2.bar(xi,     r2, width=w, label='Recall',    color='#ED7D31', **bar_kw)
    ax2.bar(xi + w, f2, width=w, label='F1-Score',  color='#70AD47', **bar_kw)
    ax2.set_xticks(xi); ax2.set_xticklabels(lbl2, fontsize=9)
    ax2.set_ylabel('Score (%)');  ax2.set_title('Precision, Recall, F1-Score Comparison')
    ax2.legend(fontsize=9); ax2.set_ylim(0, 105); ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out_png = Config.VIZ_DIR / 'seq10_all_bigru_comparison.png'
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved → {out_png}")
    return out_png


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_all()

    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD  —  BiGRU  |  seq_len=10  |  cross-class")
    print("=" * 70)
    print(f"{'Model':<12} {'Val Acc':>8} {'Precision':>10} {'Recall':>8} "
          f"{'F1':>8} {'Time':>8}")
    print("-" * 70)
    for m in sorted(results, key=lambda x: results[x]['accuracy'], reverse=True):
        r = results[m]
        print(f"{m.capitalize():<12} {r['accuracy']:>7.2f}% "
              f"{r['precision']:>9.2f}% {r['recall']:>7.2f}% "
              f"{r['f1_score']:>7.2f}% {r['training_time']:>6.1f}m")
    print("=" * 70)

    print(f"\nStep-4 Baseline (Temporal Transformer):  {STEP4_BASELINE:.2f}%")
    print(f"BiLSTM Hybrid best:                       {BILSTM_BEST:.2f}%")
    best_gru = max(results, key=lambda m: results[m]['accuracy'])
    print(f"BiGRU best ({best_gru.capitalize()}):  "
          f"{results[best_gru]['accuracy']:.2f}%")

    plot_comparison(results)
    print("\nDone.")
