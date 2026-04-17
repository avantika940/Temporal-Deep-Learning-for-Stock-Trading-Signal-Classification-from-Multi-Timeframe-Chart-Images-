"""
Train all 5 BiLSTM models for 5 epochs on the cross-class seq=10 timeline dataset
and generate a 4-panel comparison chart.
"""

import sys, os, json, time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from train import train_model, set_seed
from evaluate import evaluate_model
from dataset import create_data_loaders
from models import create_model

# ── Config shared across all runs ────────────────────────────────────────────
BASE_CONFIG = dict(
    SEQUENCE_LENGTH = 10,
    NUM_EPOCHS      = 10,
    EARLY_STOPPING  = False,   # run full 5 epochs for fair comparison
)

MODEL_TYPES = ['deep', 'attention', 'residual', 'hybrid', 'pyramidal']

# ── Train all models ─────────────────────────────────────────────────────────
def run_all():
    all_results = {}

    print("\n" + "="*80)
    print("SEQ=10 CROSS-CLASS EXPERIMENT  —  ALL 5 BiLSTM MODELS  |  5 EPOCHS EACH")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    for i, mt in enumerate(MODEL_TYPES, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/5] {mt.upper()} BiLSTM")
        print(f"{'='*70}\n")

        config = Config(MODEL_TYPE=mt, **BASE_CONFIG)
        set_seed(config.RANDOM_SEED)

        t0 = time.time()
        model, history, best_val_acc = train_model(config)
        elapsed_min = (time.time() - t0) / 60

        # Evaluate best checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, val_loader = create_data_loaders(config)
        ckpt = torch.load(config.MODELS_DIR / f'best_{mt}_bilstm.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        metrics = evaluate_model(model, val_loader, device, config)

        # evaluate_model returns values already scaled 0-100 for accuracy/precision/recall/f1
        # but as fractions 0-1 depending on version — normalise to 0-100 range
        def to_pct(v):
            return v if v > 1.5 else v * 100

        all_results[mt] = {
            'accuracy':        to_pct(metrics['accuracy']),
            'precision':       to_pct(metrics['precision']),
            'recall':          to_pct(metrics['recall']),
            'f1_score':        to_pct(metrics['f1_score']),
            'best_epoch':      ckpt['epoch'],
            'training_time':   elapsed_min,
            'epochs_trained':  5,
            'history':         history,
        }
        print(f"\n  ✓ {mt.upper()}: Val Acc = {all_results[mt]['accuracy']:.2f}%  "
              f"F1 = {all_results[mt]['f1_score']:.2f}%  "
              f"Time = {elapsed_min:.1f} min")

    # Save JSON
    out_json = config.LOGS_DIR / 'seq10_all_models_results.json'
    with open(out_json, 'w') as f:
        # history contains numpy arrays – convert
        def serialise(obj):
            if isinstance(obj, (np.integer, np.floating)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            raise TypeError
        json.dump(all_results, f, indent=2, default=serialise)
    print(f"\nResults saved → {out_json}")

    return all_results


# ── Plot 4-panel comparison ───────────────────────────────────────────────────
def plot_comparison(results):
    models_sorted = sorted(results, key=lambda m: results[m]['accuracy'])
    labels   = [m.capitalize() for m in models_sorted]
    accs     = [results[m]['accuracy']    for m in models_sorted]
    prec     = [results[m]['precision']   for m in models_sorted]
    rec      = [results[m]['recall']      for m in models_sorted]
    f1s      = [results[m]['f1_score']    for m in models_sorted]
    times    = [results[m]['training_time'] for m in models_sorted]

    BASELINE = 61.21   # Step-4 Temporal Transformer

    fig = plt.figure(figsize=(16, 6))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle(
        'BiLSTM Models — Cross-Class Timeline  |  seq_len=10\n'
        'Adani Stock Trading Signal Classification (BUY / HOLD / SELL)',
        fontsize=14, fontweight='bold', y=1.01
    )

    bar_kw = dict(edgecolor='#333', linewidth=0.8)
    BLUE   = '#4472C4'
    label_kw = dict(ha='left', va='center', fontsize=9, fontweight='bold')
    x = np.arange(len(labels))

    # ── Panel 1: Accuracy ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.barh(labels, accs, color=BLUE, **bar_kw)
    ax.axvline(BASELINE, color='red', linestyle='--', linewidth=1.5,
               label=f'Step 4 Baseline ({BASELINE}%)')
    for bar, v in zip(bars, accs):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.2f}%', **label_kw)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xlim(0, max(accs) * 1.12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    # ── Panel 2: Precision / Recall / F1 ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    # sorted by accuracy descending for this panel
    ms2  = sorted(results, key=lambda m: results[m]['accuracy'], reverse=True)
    lbl2 = [m.capitalize() for m in ms2]
    p2   = [results[m]['precision'] for m in ms2]
    r2   = [results[m]['recall']    for m in ms2]
    f2   = [results[m]['f1_score']  for m in ms2]
    xi   = np.arange(len(lbl2))
    w    = 0.25
    ax.bar(xi - w, p2, width=w, label='Precision', color='#4472C4', **bar_kw)
    ax.bar(xi,     r2, width=w, label='Recall',    color='#ED7D31', **bar_kw)
    ax.bar(xi + w, f2, width=w, label='F1-Score',  color='#70AD47', **bar_kw)
    ax.set_xticks(xi); ax.set_xticklabels(lbl2, fontsize=9)
    ax.set_ylabel('Score (%)'); ax.set_title('Precision, Recall, F1-Score Comparison')
    ax.legend(fontsize=9); ax.set_ylim(0, 105); ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / 'results' / 'visualizations' / 'seq10_all_models_comparison.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved → {out}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_all()

    print("\n" + "="*70)
    print("FINAL LEADERBOARD (seq_len=10, cross-class, 5 epochs)")
    print("="*70)
    print(f"{'Model':<12} {'Val Acc':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Time':>8}")
    print("-"*70)
    for m in sorted(results, key=lambda x: results[x]['accuracy'], reverse=True):
        r = results[m]
        print(f"{m.capitalize():<12} {r['accuracy']:>7.2f}% {r['precision']:>9.2f}% "
              f"{r['recall']:>7.2f}% {r['f1_score']:>7.2f}% {r['training_time']:>6.1f}m")
    print("="*70)

    plot_comparison(results)
    print("\nDone.")
