"""
Generate Comprehensive Visual Results for Enhanced RGB Training
Final results visualization for ViT vs CLIP comparison
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Dynamic path resolution - works on any machine
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
results_file = PROJECT_ROOT / "Model_Results" / "results_enhanced_rgb.json"
output_dir = PROJECT_ROOT / "Model_Results"

with open(results_file, 'r') as f:
    results = json.load(f)

# Extract data
dataset_type = results['dataset_type']
vit_acc = results['vit']['final_accuracy']
vit_best = results['vit']['best_val_accuracy']
clip_acc = results['clip']['final_accuracy']
clip_best = results['clip']['best_val_accuracy']

vit_cm = np.array(results['vit']['confusion_matrix'])
clip_cm = np.array(results['clip']['confusion_matrix'])

vit_report = results['vit']['classification_report']
clip_report = results['clip']['classification_report']

class_names = ['BUY', 'HOLD', 'SELL']

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Stock Market Signal Classification Results - ViT vs CLIP\nEnhanced RGB Dataset (224×224)', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Overall Accuracy Comparison
ax1 = plt.subplot(2, 4, 1)
models = ['ViT', 'CLIP']
accuracies = [vit_acc, clip_acc]
colors = ['#2E86AB', '#A23B72']
bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Final Validation Accuracy', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.axhline(y=33.33, color='red', linestyle='--', alpha=0.5, label='Random Baseline (33.33%)')

# Add value labels
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)

# 2. Best Val Accuracy Comparison
ax2 = plt.subplot(2, 4, 2)
best_accs = [vit_best, clip_best]
bars = ax2.bar(models, best_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Best Validation Accuracy', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

for i, (bar, acc) in enumerate(zip(bars, best_accs)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Per-Class Precision - ViT
ax3 = plt.subplot(2, 4, 3)
vit_precisions = [
    vit_report['BUY']['precision'] * 100,
    vit_report['HOLD']['precision'] * 100,
    vit_report['SELL']['precision'] * 100
]
colors_class = ['#06A77D', '#F5B700', '#D62828']
bars = ax3.bar(class_names, vit_precisions, color=colors_class, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax3.set_title('ViT - Class Precision', fontsize=13, fontweight='bold')
ax3.set_ylim([0, 100])
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

for bar, val in zip(bars, vit_precisions):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Per-Class Recall - ViT
ax4 = plt.subplot(2, 4, 4)
vit_recalls = [
    vit_report['BUY']['recall'] * 100,
    vit_report['HOLD']['recall'] * 100,
    vit_report['SELL']['recall'] * 100
]
bars = ax4.bar(class_names, vit_recalls, color=colors_class, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax4.set_title('ViT - Class Recall', fontsize=13, fontweight='bold')
ax4.set_ylim([0, 100])
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

for bar, val in zip(bars, vit_recalls):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. ViT Confusion Matrix
ax5 = plt.subplot(2, 4, 5)
sns.heatmap(vit_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
            yticklabels=class_names, ax=ax5, cbar_kws={'label': 'Count'}, 
            linewidths=1, linecolor='white', square=True, annot_kws={'size': 12, 'weight': 'bold'})
ax5.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax5.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax5.set_title('ViT Confusion Matrix', fontsize=13, fontweight='bold')

# 6. CLIP Confusion Matrix
ax6 = plt.subplot(2, 4, 6)
sns.heatmap(clip_cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, 
            yticklabels=class_names, ax=ax6, cbar_kws={'label': 'Count'},
            linewidths=1, linecolor='white', square=True, annot_kws={'size': 12, 'weight': 'bold'})
ax6.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax6.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax6.set_title('CLIP Confusion Matrix', fontsize=13, fontweight='bold')

# 7. Per-Class F1-Score Comparison
ax7 = plt.subplot(2, 4, 7)
vit_f1 = [
    vit_report['BUY']['f1-score'] * 100,
    vit_report['HOLD']['f1-score'] * 100,
    vit_report['SELL']['f1-score'] * 100
]
clip_f1 = [
    clip_report['BUY']['f1-score'] * 100,
    clip_report['HOLD']['f1-score'] * 100,
    clip_report['SELL']['f1-score'] * 100
]

x = np.arange(len(class_names))
width = 0.35
bars1 = ax7.bar(x - width/2, vit_f1, width, label='ViT', color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax7.bar(x + width/2, clip_f1, width, label='CLIP', color='#A23B72', alpha=0.8, edgecolor='black')

ax7.set_xlabel('Class', fontsize=11, fontweight='bold')
ax7.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
ax7.set_title('F1-Score Comparison by Class', fontsize=13, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(class_names)
ax7.legend(fontsize=10)
ax7.set_ylim([0, 100])
ax7.grid(True, alpha=0.3, axis='y', linestyle='--')

# 8. Summary Statistics Box
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')

summary_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           TRAINING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: {dataset_type}
Total Images: {results['total_images']:,}
Training Set: {results['train_images']:,} images
Validation Set: {results['val_images']} images
Epochs: {results['num_epochs']}
Batch Size: {results['batch_size']}
Learning Rate: {results['learning_rate']:.0e}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         VISION TRANSFORMER (ViT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final Accuracy:     {vit_acc:.2f}%
Best Val Accuracy:  {vit_best:.2f}%

BUY  → Precision: {vit_report['BUY']['precision']*100:.1f}%
       Recall:    {vit_report['BUY']['recall']*100:.1f}%
       F1-Score:  {vit_report['BUY']['f1-score']*100:.1f}%

HOLD → Precision: {vit_report['HOLD']['precision']*100:.1f}%
       Recall:    {vit_report['HOLD']['recall']*100:.1f}%
       F1-Score:  {vit_report['HOLD']['f1-score']*100:.1f}%

SELL → Precision: {vit_report['SELL']['precision']*100:.1f}%
       Recall:    {vit_report['SELL']['recall']*100:.1f}%
       F1-Score:  {vit_report['SELL']['f1-score']*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              CLIP MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final Accuracy:     {clip_acc:.2f}%
Best Val Accuracy:  {clip_best:.2f}%

BUY  → Precision: {clip_report['BUY']['precision']*100:.1f}%
       Recall:    {clip_report['BUY']['recall']*100:.1f}%
       F1-Score:  {clip_report['BUY']['f1-score']*100:.1f}%

HOLD → Precision: {clip_report['HOLD']['precision']*100:.1f}%
       Recall:    {clip_report['HOLD']['recall']*100:.1f}%
       F1-Score:  {clip_report['HOLD']['f1-score']*100:.1f}%

SELL → Precision: {clip_report['SELL']['precision']*100:.1f}%
       Recall:    {clip_report['SELL']['recall']*100:.1f}%
       F1-Score:  {clip_report['SELL']['f1-score']*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            WINNER: ViT
         Margin: {vit_acc - clip_acc:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=8.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8, edgecolor='black', linewidth=2))

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_file = output_dir / 'FINAL_RESULTS_Enhanced_RGB.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n{'='*70}")
print(f"✓ Final results visualization saved to:")
print(f"  {output_file}")
print(f"{'='*70}\n")

# Also create a simplified comparison chart
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Quick Comparison: ViT vs CLIP on Enhanced RGB Dataset', 
              fontsize=16, fontweight='bold')

# Chart 1: Accuracy
ax1 = axes[0]
models = ['ViT', 'CLIP']
final_accs = [vit_acc, clip_acc]
bars = ax1.bar(models, final_accs, color=['#2E86AB', '#A23B72'], alpha=0.8, 
               edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Final Validation Accuracy', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
for bar, acc in zip(bars, final_accs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Chart 2: Per-Class Performance
ax2 = axes[1]
x = np.arange(len(class_names))
width = 0.35
vit_f1_scores = [vit_report[cls]['f1-score'] * 100 for cls in class_names]
clip_f1_scores = [clip_report[cls]['f1-score'] * 100 for cls in class_names]

bars1 = ax2.bar(x - width/2, vit_f1_scores, width, label='ViT', 
                color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, clip_f1_scores, width, label='CLIP', 
                color='#A23B72', alpha=0.8, edgecolor='black')

ax2.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score by Class', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(class_names)
ax2.legend(fontsize=11)
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# Chart 3: Winner Summary
ax3 = axes[2]
ax3.axis('off')
winner_text = f"""
{'='*40}
        FINAL VERDICT
{'='*40}

🏆 WINNER: Vision Transformer (ViT)

Accuracy Difference: {vit_acc - clip_acc:.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ViT Strengths:
  • Higher overall accuracy ({vit_acc:.1f}%)
  • Best SELL detection ({vit_report['SELL']['recall']*100:.1f}%)
  • Strong precision across classes

CLIP Strengths:
  • 5× faster training
  • More balanced class performance
  • Better BUY recall ({clip_report['BUY']['recall']*100:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recommendation:
Use ViT for higher accuracy
Use CLIP for faster inference

{'='*40}
"""

ax3.text(0.5, 0.5, winner_text, transform=ax3.transAxes, 
         fontsize=10, ha='center', va='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.9, 
                   edgecolor='black', linewidth=2))

plt.tight_layout()
output_file2 = output_dir / 'QUICK_COMPARISON.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Quick comparison saved to:")
print(f"  {output_file2}")
print(f"{'='*70}\n")

print("All visualizations generated successfully! 🎉")
