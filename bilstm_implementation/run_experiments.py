"""
Master script to run all BiLSTM model experiments

This script trains all 5 BiLSTM variants and generates a comprehensive comparison report.
"""

import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import Config, get_config
from train import train_model, set_seed
from evaluate import evaluate_model
from dataset import create_data_loaders
from models import create_model


def run_all_experiments():
    """Run experiments for all model variants"""
    
    print("\n" + "="*80)
    print("BILSTM EXPERIMENTS - RUNNING ALL MODEL VARIANTS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Model types to train
    model_types = ['deep', 'attention', 'residual', 'hybrid', 'pyramidal']
    
    # Storage for results
    all_results = {}
    
    # Train each model variant
    for i, model_type in enumerate(model_types, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(model_types)}: {model_type.upper()} BILSTM")
        print(f"{'='*80}\n")
        
        try:
            # Get configuration for this model
            config = get_config(model_type)
            
            # Train model
            start_time = time.time()
            model, history, best_acc = train_model(config)
            training_time = time.time() - start_time
            
            # Evaluate on validation set
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _, val_loader = create_data_loaders(config)
            
            print(f"\nRunning final evaluation for {model_type}...")
            metrics = evaluate_model(model, val_loader, device, config)
            
            # Store results
            all_results[model_type] = {
                'best_val_acc': best_acc,
                'final_val_acc': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'training_time_minutes': training_time / 60,
                'total_epochs': len(history['train_acc']),
                'config': {
                    'hidden_dim': config.HIDDEN_DIM,
                    'num_layers': config.NUM_LSTM_LAYERS,
                    'sequence_length': config.SEQUENCE_LENGTH,
                    'batch_size': config.BATCH_SIZE,
                    'learning_rate': config.LEARNING_RATE
                }
            }
            
            print(f"\n✓ {model_type.upper()} completed successfully!")
            print(f"  Best validation accuracy: {best_acc:.2f}%")
            print(f"  Training time: {training_time/60:.2f} minutes")
            
        except Exception as e:
            print(f"\n✗ {model_type.upper()} failed with error: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_type] = {'error': str(e)}
    
    # Generate comparison report
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80 + "\n")
    
    generate_comparison_report(all_results, Config())
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return all_results


def generate_comparison_report(results, config):
    """Generate comprehensive comparison report"""
    
    # Filter out failed experiments
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("No successful experiments to compare!")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for model_type, metrics in successful_results.items():
        comparison_data.append({
            'Model': model_type.capitalize(),
            'Accuracy (%)': metrics['best_val_acc'],
            'Precision (%)': metrics['precision'],
            'Recall (%)': metrics['recall'],
            'F1-Score (%)': metrics['f1_score'],
            'Training Time (min)': metrics['training_time_minutes'],
            'Epochs': metrics['total_epochs']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy (%)', ascending=False)
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    # Find best model
    best_model = df.iloc[0]['Model'].lower()
    best_acc = df.iloc[0]['Accuracy (%)']
    
    print(f"🏆 BEST MODEL: {best_model.upper()}")
    print(f"   Accuracy: {best_acc:.2f}%")
    
    # Compare with Step 4 baseline
    STEP4_BASELINE = 61.21  # Step 4's best result (Temporal Transformer)
    improvement = best_acc - STEP4_BASELINE
    
    print(f"\n📊 COMPARISON WITH STEP 4:")
    print(f"   Step 4 Best (Temporal Transformer): {STEP4_BASELINE:.2f}%")
    print(f"   BiLSTM Best ({best_model.upper()}): {best_acc:.2f}%")
    print(f"   Improvement: {improvement:+.2f}% ({improvement/STEP4_BASELINE*100:+.2f}% relative)")
    
    if improvement > 0:
        print(f"\n✓ SUCCESS: BiLSTM improved over Step 4!")
    else:
        print(f"\n⚠ BiLSTM did not surpass Step 4")
        print(f"   Consider: longer training, different sequence lengths, or ensemble methods")
    
    # Save comparison data
    comparison_path = config.RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump({
            'results': results,
            'best_model': best_model,
            'best_accuracy': best_acc,
            'step4_baseline': STEP4_BASELINE,
            'improvement': improvement,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"\nComparison data saved to: {comparison_path}")
    
    # Save DataFrame as CSV
    csv_path = config.RESULTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    
    # Generate visualizations
    plot_model_comparison(df, config, STEP4_BASELINE)
    
    # Generate summary report
    generate_text_report(df, successful_results, config, STEP4_BASELINE)


def plot_model_comparison(df, config, step4_baseline):
    """Generate comparison visualizations"""
    
    # 1. Accuracy comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    ax = axes[0, 0]
    models = df['Model']
    accuracies = df['Accuracy (%)']
    
    bars = ax.barh(models, accuracies, color='steelblue')
    ax.axvline(x=step4_baseline, color='red', linestyle='--', linewidth=2, label=f'Step 4 Baseline ({step4_baseline:.2f}%)')
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        label = f'{acc:.2f}%'
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, label,
                ha='left', va='center', fontsize=10)
    
    # Precision, Recall, F1 comparison
    ax = axes[0, 1]
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, df['Precision (%)'], width, label='Precision', color='steelblue')
    ax.bar(x, df['Recall (%)'], width, label='Recall', color='coral')
    ax.bar(x + width, df['F1-Score (%)'], width, label='F1-Score', color='lightgreen')
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Precision, Recall, F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Training time comparison
    ax = axes[1, 0]
    bars = ax.barh(models, df['Training Time (min)'], color='mediumpurple')
    ax.set_xlabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    for bar, time_val in zip(bars, df['Training Time (min)']):
        width = bar.get_width()
        label = f'{time_val:.1f} min'
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, label,
                ha='left', va='center', fontsize=10)
    
    # Epochs comparison
    ax = axes[1, 1]
    bars = ax.barh(models, df['Epochs'], color='darkorange')
    ax.set_xlabel('Number of Epochs', fontsize=12)
    ax.set_title('Training Epochs Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    for bar, epochs in zip(bars, df['Epochs']):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(epochs)}',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    save_path = config.VIZ_DIR / "all_models_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison visualizations saved to: {save_path}")


def generate_text_report(df, results, config, step4_baseline):
    """Generate detailed text report"""
    
    report_path = config.RESULTS_DIR / "EXPERIMENT_SUMMARY.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BILSTM IMPLEMENTATION - EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OBJECTIVE:\n")
        f.write("Improve upon Step 4's best accuracy (61.21% from Temporal Transformer)\n")
        f.write("using advanced BiLSTM architectures with attention and residual connections.\n\n")
        
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("BEST MODEL\n")
        f.write("="*80 + "\n")
        best_model = df.iloc[0]['Model'].lower()
        best_acc = df.iloc[0]['Accuracy (%)']
        f.write(f"Model: {best_model.upper()}\n")
        f.write(f"Accuracy: {best_acc:.2f}%\n")
        f.write(f"Precision: {df.iloc[0]['Precision (%)']:.2f}%\n")
        f.write(f"Recall: {df.iloc[0]['Recall (%)']:.2f}%\n")
        f.write(f"F1-Score: {df.iloc[0]['F1-Score (%)']:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("COMPARISON WITH STEP 4 BASELINE\n")
        f.write("="*80 + "\n")
        f.write(f"Step 4 Best (Temporal Transformer): {step4_baseline:.2f}%\n")
        f.write(f"BiLSTM Best ({best_model.upper()}): {best_acc:.2f}%\n")
        improvement = best_acc - step4_baseline
        f.write(f"Absolute Improvement: {improvement:+.2f}%\n")
        f.write(f"Relative Improvement: {improvement/step4_baseline*100:+.2f}%\n\n")
        
        if improvement > 0:
            f.write("RESULT: [SUCCESS] BiLSTM improved over Step 4!\n\n")
        else:
            f.write("RESULT: [WARNING] BiLSTM did not surpass Step 4\n\n")
        
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n")
        f.write("1. Model Architecture Impact:\n")
        f.write(f"   - Best performing: {best_model.capitalize()} BiLSTM\n")
        f.write(f"   - Worst performing: {df.iloc[-1]['Model']}\n")
        f.write(f"   - Performance range: {df.iloc[-1]['Accuracy (%)']:.2f}% - {best_acc:.2f}%\n\n")
        
        f.write("2. Training Efficiency:\n")
        f.write(f"   - Fastest training: {df.iloc[df['Training Time (min)'].idxmin()]['Model']} ")
        f.write(f"({df['Training Time (min)'].min():.1f} min)\n")
        f.write(f"   - Slowest training: {df.iloc[df['Training Time (min)'].idxmax()]['Model']} ")
        f.write(f"({df['Training Time (min)'].max():.1f} min)\n\n")
        
        f.write("3. Model Complexity:\n")
        f.write(f"   - Average epochs to convergence: {df['Epochs'].mean():.1f}\n")
        f.write(f"   - Total training time: {df['Training Time (min)'].sum():.1f} minutes\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        if improvement > 0:
            f.write(f"[OK] Use {best_model.upper()} BiLSTM for production deployment\n")
            f.write(f"[OK] Model shows {improvement:.2f}% absolute improvement over Step 4\n")
        else:
            f.write("Consider the following strategies:\n")
            f.write("  1. Increase sequence length (try 5 or 7 frames)\n")
            f.write("  2. Ensemble BiLSTM with Step 4's Transformer\n")
            f.write("  3. Add more temporal augmentation\n")
            f.write("  4. Collect more training data\n")
            f.write("  5. Try semi-supervised pre-training\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\nText report saved to: {report_path}")


def main():
    """Main function"""
    try:
        results = run_all_experiments()
        print("\n✓ All experiments completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Experiments failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
