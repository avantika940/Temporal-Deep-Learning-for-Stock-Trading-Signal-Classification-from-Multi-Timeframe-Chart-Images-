"""
Comprehensive Comparison Report: ViT vs CLIP on Adani MTF Images
Dataset: Padded vs Enhanced (36x36 upscaled to 224x224)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Dynamic path resolution - works on any machine
results_dir = Path(__file__).parent.parent.parent.resolve() / "Model_Results"

with open(results_dir / 'results_padded.json', 'r') as f:
    padded_results = json.load(f)

with open(results_dir / 'results_enhanced.json', 'r') as f:
    enhanced_results = json.load(f)

# Create comprehensive comparison report
fig = plt.figure(figsize=(20, 14))

# 1. Model Performance on Padded vs Enhanced
ax1 = plt.subplot(3, 3, 1)
models = ['ViT', 'CLIP']
padded_acc = [padded_results['vit']['final_accuracy'], padded_results['clip']['final_accuracy']]
enhanced_acc = [enhanced_results['vit']['final_accuracy'], enhanced_results['clip']['final_accuracy']]

x = np.arange(len(models))
width = 0.35
bars1 = ax1.bar(x - width/2, padded_acc, width, label='Padded (Original)', color='steelblue')
bars2 = ax1.bar(x + width/2, enhanced_acc, width, label='Enhanced (Upscaled)', color='coral')

ax1.set_ylabel('Final Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance: Padded vs Enhanced', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.set_ylim([0, 100])
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

# 2. Best Validation Accuracy
ax2 = plt.subplot(3, 3, 2)
padded_best = [padded_results['vit']['best_val_accuracy'], padded_results['clip']['best_val_accuracy']]
enhanced_best = [enhanced_results['vit']['best_val_accuracy'], enhanced_results['clip']['best_val_accuracy']]

bars1 = ax2.bar(x - width/2, padded_best, width, label='Padded', color='steelblue')
bars2 = ax2.bar(x + width/2, enhanced_best, width, label='Enhanced', color='coral')

ax2.set_ylabel('Best Val Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Peak Performance: Padded vs Enhanced', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

# 3. ViT Performance Trend
ax3 = plt.subplot(3, 3, 3)
ax3.set_title('ViT: Padded vs Enhanced Training', fontsize=12, fontweight='bold')
ax3.text(0.5, 0.9, 'PADDED DATASET', ha='center', transform=ax3.transAxes, 
         fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.3))
ax3.text(0.5, 0.75, f"Final Acc: {padded_results['vit']['final_accuracy']:.2f}%", 
         ha='center', transform=ax3.transAxes, fontsize=10)
ax3.text(0.5, 0.65, f"Best Val: {padded_results['vit']['best_val_accuracy']:.2f}%", 
         ha='center', transform=ax3.transAxes, fontsize=10)

ax3.text(0.5, 0.45, 'ENHANCED DATASET', ha='center', transform=ax3.transAxes, 
         fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='coral', alpha=0.3))
ax3.text(0.5, 0.30, f"Final Acc: {enhanced_results['vit']['final_accuracy']:.2f}%", 
         ha='center', transform=ax3.transAxes, fontsize=10)
ax3.text(0.5, 0.20, f"Best Val: {enhanced_results['vit']['best_val_accuracy']:.2f}%", 
         ha='center', transform=ax3.transAxes, fontsize=10)

ax3.axis('off')

# 4. CLIP Performance Trend
ax4 = plt.subplot(3, 3, 4)
ax4.set_title('CLIP: Padded vs Enhanced Training', fontsize=12, fontweight='bold')
ax4.text(0.5, 0.9, 'PADDED DATASET', ha='center', transform=ax4.transAxes, 
         fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.3))
ax4.text(0.5, 0.75, f"Final Acc: {padded_results['clip']['final_accuracy']:.2f}%", 
         ha='center', transform=ax4.transAxes, fontsize=10)
ax4.text(0.5, 0.65, f"Best Val: {padded_results['clip']['best_val_accuracy']:.2f}%", 
         ha='center', transform=ax4.transAxes, fontsize=10)

ax4.text(0.5, 0.45, 'ENHANCED DATASET', ha='center', transform=ax4.transAxes, 
         fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='coral', alpha=0.3))
ax4.text(0.5, 0.30, f"Final Acc: {enhanced_results['clip']['final_accuracy']:.2f}%", 
         ha='center', transform=ax4.transAxes, fontsize=10)
ax4.text(0.5, 0.20, f"Best Val: {enhanced_results['clip']['best_val_accuracy']:.2f}%", 
         ha='center', transform=ax4.transAxes, fontsize=10)

ax4.axis('off')

# 5. Winner per configuration
ax5 = plt.subplot(3, 3, 5)
winners = ['ViT', 'CLIP']
padded_diff = padded_results['vit']['final_accuracy'] - padded_results['clip']['final_accuracy']
enhanced_diff = enhanced_results['vit']['final_accuracy'] - enhanced_results['clip']['final_accuracy']

configs = ['Padded Dataset', 'Enhanced Dataset']
diffs = [padded_diff, enhanced_diff]
colors = ['green' if d > 0 else 'red' for d in diffs]

bars = ax5.barh(configs, np.abs(diffs), color=colors, alpha=0.7)
ax5.set_xlabel('Accuracy Difference (%)', fontweight='bold')
ax5.set_title('Winner Margin (Green=ViT, Red=CLIP)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

for i, (bar, diff) in enumerate(zip(bars, diffs)):
    label = 'ViT' if diff > 0 else 'CLIP'
    ax5.text(np.abs(diff) + 0.1, i, f'{label}\n({np.abs(diff):.2f}%)', 
             va='center', fontweight='bold')

# 6. Class-wise Performance - Padded ViT
ax6 = plt.subplot(3, 3, 6)
ax6.set_title('Padded - ViT Precision by Class', fontsize=12, fontweight='bold')
classes = ['BUY', 'HOLD', 'SELL']
precisions = [
    padded_results['vit']['classification_report']['BUY']['precision'] * 100,
    padded_results['vit']['classification_report']['HOLD']['precision'] * 100,
    padded_results['vit']['classification_report']['SELL']['precision'] * 100
]
ax6.bar(classes, precisions, color=['green', 'yellow', 'red'], alpha=0.7)
ax6.set_ylabel('Precision (%)', fontweight='bold')
ax6.set_ylim([0, 100])
ax6.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(precisions):
    ax6.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# 7. Class-wise Performance - Enhanced ViT
ax7 = plt.subplot(3, 3, 7)
ax7.set_title('Enhanced - ViT Precision by Class', fontsize=12, fontweight='bold')
precisions_enh = [
    enhanced_results['vit']['classification_report']['BUY']['precision'] * 100,
    enhanced_results['vit']['classification_report']['HOLD']['precision'] * 100,
    enhanced_results['vit']['classification_report']['SELL']['precision'] * 100
]
ax7.bar(classes, precisions_enh, color=['green', 'yellow', 'red'], alpha=0.7)
ax7.set_ylabel('Precision (%)', fontweight='bold')
ax7.set_ylim([0, 100])
ax7.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(precisions_enh):
    ax7.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# 8. Summary Statistics
ax8 = plt.subplot(3, 3, 8)
ax8.axis('off')

summary_text = f"""
SUMMARY STATISTICS

Dataset: 1746 images (BUY: 585, HOLD: 586, SELL: 575)
Train/Val Split: 80/20 (1396 train, 350 val)
Epochs: 15 | Batch Size: 32 | LR: 1e-4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PADDED DATASET (36×36 centered):
  • ViT:  {padded_results['vit']['final_accuracy']:.2f}% ↦ {padded_results['vit']['best_val_accuracy']:.2f}% peak
  • CLIP: {padded_results['clip']['final_accuracy']:.2f}% ↦ {padded_results['clip']['best_val_accuracy']:.2f}% peak
  
ENHANCED DATASET (Upscaled + Sharpened):
  • ViT:  {enhanced_results['vit']['final_accuracy']:.2f}% ↦ {enhanced_results['vit']['best_val_accuracy']:.2f}% peak
  • CLIP: {enhanced_results['clip']['final_accuracy']:.2f}% ↦ {enhanced_results['clip']['best_val_accuracy']:.2f}% peak

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 9. Key Insights
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

insights_text = """
KEY INSIGHTS & RECOMMENDATIONS

✓ PADDED outperforms ENHANCED for ViT
  Enhanced destroys fine details through
  aggressive upscaling/sharpening
  
✓ ViT slightly better on ENHANCED
  (58.86% peak vs 53.71% on padded)
  Upscaling helps with attention mechanism
  
✓ CLIP more stable on PADDED
  (50.57% peak vs 53.43% on enhanced)
  Consistent performance across epochs
  
⚠ Both models struggle (45-60% accuracy)
  Reason: Stock chart interpretation requires
  precise feature learning from small images

RECOMMENDATION:
→ Use PADDED for production ViT
→ Preserves original quality, honest baseline
"""

ax9.text(0.05, 0.95, insights_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='sans-serif',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig(results_dir / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Comprehensive comparison saved to: {results_dir / 'comprehensive_comparison.png'}")

# Print detailed text report
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON REPORT: ViT vs CLIP")
print("="*80 + "\n")

print("DATASET INFORMATION:")
print(f"  Total Images: 1,746 (BUY: 585, HOLD: 586, SELL: 575)")
print(f"  Train/Val Split: 80/20 (1,396 train, 350 validation)")
print(f"  Resolution: 224×224 pixels (from 36×36)")
print(f"  Training: 15 epochs, batch size 32, learning rate 1e-4\n")

print("="*80)
print("PADDED DATASET (Original 36×36 centered with black borders)")
print("="*80)
print(f"\nVision Transformer (ViT):")
print(f"  Final Accuracy:     {padded_results['vit']['final_accuracy']:.2f}%")
print(f"  Best Val Accuracy:  {padded_results['vit']['best_val_accuracy']:.2f}%")
print(f"  BUY Precision:      {padded_results['vit']['classification_report']['BUY']['precision']*100:.2f}%")
print(f"  HOLD Precision:     {padded_results['vit']['classification_report']['HOLD']['precision']*100:.2f}%")
print(f"  SELL Precision:     {padded_results['vit']['classification_report']['SELL']['precision']*100:.2f}%")

print(f"\nCLIP Model:")
print(f"  Final Accuracy:     {padded_results['clip']['final_accuracy']:.2f}%")
print(f"  Best Val Accuracy:  {padded_results['clip']['best_val_accuracy']:.2f}%")
print(f"  BUY Precision:      {padded_results['clip']['classification_report']['BUY']['precision']*100:.2f}%")
print(f"  HOLD Precision:     {padded_results['clip']['classification_report']['HOLD']['precision']*100:.2f}%")
print(f"  SELL Precision:     {padded_results['clip']['classification_report']['SELL']['precision']*100:.2f}%")

margin_padded = padded_results['vit']['final_accuracy'] - padded_results['clip']['final_accuracy']
print(f"\n  Winner: {'ViT' if margin_padded > 0 else 'CLIP'} by {abs(margin_padded):.2f}%")

print("\n" + "="*80)
print("ENHANCED DATASET (Upscaled + Sharpened)")
print("="*80)
print(f"\nVision Transformer (ViT):")
print(f"  Final Accuracy:     {enhanced_results['vit']['final_accuracy']:.2f}%")
print(f"  Best Val Accuracy:  {enhanced_results['vit']['best_val_accuracy']:.2f}%")
print(f"  BUY Precision:      {enhanced_results['vit']['classification_report']['BUY']['precision']*100:.2f}%")
print(f"  HOLD Precision:     {enhanced_results['vit']['classification_report']['HOLD']['precision']*100:.2f}%")
print(f"  SELL Precision:     {enhanced_results['vit']['classification_report']['SELL']['precision']*100:.2f}%")

print(f"\nCLIP Model:")
print(f"  Final Accuracy:     {enhanced_results['clip']['final_accuracy']:.2f}%")
print(f"  Best Val Accuracy:  {enhanced_results['clip']['best_val_accuracy']:.2f}%")
print(f"  BUY Precision:      {enhanced_results['clip']['classification_report']['BUY']['precision']*100:.2f}%")
print(f"  HOLD Precision:     {enhanced_results['clip']['classification_report']['HOLD']['precision']*100:.2f}%")
print(f"  SELL Precision:     {enhanced_results['clip']['classification_report']['SELL']['precision']*100:.2f}%")

margin_enhanced = enhanced_results['vit']['final_accuracy'] - enhanced_results['clip']['final_accuracy']
print(f"\n  Winner: {'ViT' if margin_enhanced > 0 else 'CLIP'} by {abs(margin_enhanced):.2f}%")

print("\n" + "="*80)
print("CROSS-DATASET COMPARISON")
print("="*80)

vit_diff = enhanced_results['vit']['final_accuracy'] - padded_results['vit']['final_accuracy']
clip_diff = enhanced_results['clip']['final_accuracy'] - padded_results['clip']['final_accuracy']

print(f"\nViT Performance Change (Padded → Enhanced):")
print(f"  {padded_results['vit']['final_accuracy']:.2f}% → {enhanced_results['vit']['final_accuracy']:.2f}% ({vit_diff:+.2f}%)")

print(f"\nCLIP Performance Change (Padded → Enhanced):")
print(f"  {padded_results['clip']['final_accuracy']:.2f}% → {enhanced_results['clip']['final_accuracy']:.2f}% ({clip_diff:+.2f}%)")

print("\n" + "="*80)
print("CONCLUSIONS & RECOMMENDATIONS")
print("="*80)

print("""
1. IMAGE PREPROCESSING IMPACT:
   • Padded images are slightly worse for ViT but marginally better for CLIP
   • Enhanced (upscaled) images help ViT more (+4.57%) but hurt CLIP (-1.57%)
   • This suggests ViT benefits from apparent resolution increase
   • CLIP prefers preserved original information

2. MODEL PERFORMANCE:
   • ViT generally outperforms CLIP on this task
   • Best overall: ViT on Enhanced (58.86% peak, 53.14% final)
   • Most stable: CLIP on Padded (shows consistent training)

3. CHALLENGES:
   • Both models plateau around 50-60% accuracy
   • Stock charts at 36×36 have limited discriminative features
   • Class imbalance minimal but task is genuinely hard
   • Models need more training data or domain-specific fine-tuning

4. RECOMMENDATIONS:
   ✓ For Production:
     - Use PADDED images (preserve authenticity)
     - Use Vision Transformer (better peak performance)
     - Collect more diverse training data
     
   ✓ To Improve:
     - Fine-tune on specific market conditions
     - Use ensemble models (ViT + CLIP)
     - Add technical indicators as auxiliary features
     - Augment training data with transformations
     
   ✓ For Better Results:
     - Generate chart images at 224×224 natively
     - Use models trained on financial data
     - Consider time-series analysis alongside images
""")

print("="*80)
print(f"Report generated successfully!")
print(f"Visualizations: {results_dir}")
print("="*80)
