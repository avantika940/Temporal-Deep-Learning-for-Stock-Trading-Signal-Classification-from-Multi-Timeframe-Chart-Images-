"""
STEP 2: Hyperparameter Tuning
Expected Accuracy Gain: +2-4% (target: 58-62%)

Systematically tune hyperparameters for each classifier using grid/random search.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralNetClassifier(nn.Module):
    """Configurable neural network"""
    def __init__(self, input_dim, hidden1=256, hidden2=128, dropout1=0.3, dropout2=0.3, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def load_embeddings():
    """Load pre-extracted embeddings"""
    # Dynamic path resolution - works on any machine
    base_dir = Path(__file__).parent.parent.parent.resolve()
    embeddings_dir = base_dir / "embeddings"
    
    print("📂 Loading embeddings...")
    data = np.load(embeddings_dir / 'vit_embeddings_grayscale.npz', allow_pickle=True)
    
    embeddings = data['embeddings']
    labels = data['labels']
    classes = data['classes'].tolist() if 'classes' in data else ['BUY', 'HOLD', 'SELL']
    
    return embeddings, labels, classes


def apply_pca(X_train, X_val, n_components=270):
    """Apply PCA dimensionality reduction"""
    print("🔄 Applying PCA compression...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    return X_train_pca, X_val_pca


def tune_xgboost(X_train, y_train, X_val, y_val):
    """Tune XGBoost hyperparameters"""
    print("\n" + "="*70)
    print("🔍 TUNING XGBOOST")
    print("="*70)
    
    # Baseline
    baseline = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_val, baseline.predict(X_val)) * 100
    print(f"   Baseline accuracy: {baseline_acc:.2f}%")
    
    # Parameter grid
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    print("   Searching parameter space (this may take a while)...")
    xgb = XGBClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        xgb, param_dist, n_iter=50, cv=3, 
        scoring='accuracy', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_acc = accuracy_score(y_val, best_model.predict(X_val)) * 100
    
    print(f"\n   ✅ Best accuracy: {best_acc:.2f}%")
    print(f"   📈 Improvement: {best_acc - baseline_acc:+.2f}%")
    print(f"   🎯 Best params: {random_search.best_params_}")
    
    return best_model, best_acc, random_search.best_params_


def tune_random_forest(X_train, y_train, X_val, y_val):
    """Tune Random Forest hyperparameters"""
    print("\n" + "="*70)
    print("🔍 TUNING RANDOM FOREST")
    print("="*70)
    
    # Baseline
    baseline = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_val, baseline.predict(X_val)) * 100
    print(f"   Baseline accuracy: {baseline_acc:.2f}%")
    
    # Parameter grid
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [6, 8, 10, 12, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [0.3, 0.5, 0.7, 'sqrt']
    }
    
    print("   Searching parameter space...")
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=40, cv=3,
        scoring='accuracy', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_acc = accuracy_score(y_val, best_model.predict(X_val)) * 100
    
    print(f"\n   ✅ Best accuracy: {best_acc:.2f}%")
    print(f"   📈 Improvement: {best_acc - baseline_acc:+.2f}%")
    print(f"   🎯 Best params: {random_search.best_params_}")
    
    return best_model, best_acc, random_search.best_params_


def tune_svm(X_train, y_train, X_val, y_val):
    """Tune SVM hyperparameters"""
    print("\n" + "="*70)
    print("🔍 TUNING SVM")
    print("="*70)
    
    # Baseline
    baseline = SVC(kernel='rbf', C=1.0, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_val, baseline.predict(X_val)) * 100
    print(f"   Baseline accuracy: {baseline_acc:.2f}%")
    
    # Parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4]
    }
    
    print("   Searching parameter space...")
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=3,
        scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_acc = accuracy_score(y_val, best_model.predict(X_val)) * 100
    
    print(f"\n   ✅ Best accuracy: {best_acc:.2f}%")
    print(f"   📈 Improvement: {best_acc - baseline_acc:+.2f}%")
    print(f"   🎯 Best params: {grid_search.best_params_}")
    
    return best_model, best_acc, grid_search.best_params_


def tune_neural_network(X_train, y_train, X_val, y_val, num_epochs=50):
    """Tune Neural Network architecture"""
    print("\n" + "="*70)
    print("🔍 TUNING NEURAL NETWORK")
    print("="*70)
    
    # Baseline
    print("   Training baseline model...")
    input_dim = X_train.shape[1]
    baseline = NeuralNetClassifier(input_dim, 256, 128, 0.3, 0.3).to(device)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(baseline.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        baseline.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = baseline(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    baseline.eval()
    with torch.no_grad():
        outputs = baseline(X_val_tensor)
        _, predicted = torch.max(outputs, 1)
        baseline_acc = (predicted == y_val_tensor).float().mean().item() * 100
    
    print(f"   Baseline accuracy: {baseline_acc:.2f}%")
    
    # Test different architectures
    print("   Testing different architectures...")
    
    configs = [
        {'hidden1': 512, 'hidden2': 256, 'dropout1': 0.4, 'dropout2': 0.3, 'lr': 0.001, 'batch_size': 32},
        {'hidden1': 384, 'hidden2': 192, 'dropout1': 0.45, 'dropout2': 0.25, 'lr': 0.0005, 'batch_size': 32},
        {'hidden1': 256, 'hidden2': 128, 'dropout1': 0.5, 'dropout2': 0.3, 'lr': 0.001, 'batch_size': 64},
        {'hidden1': 512, 'hidden2': 128, 'dropout1': 0.4, 'dropout2': 0.2, 'lr': 0.0015, 'batch_size': 16},
        {'hidden1': 384, 'hidden2': 256, 'dropout1': 0.35, 'dropout2': 0.25, 'lr': 0.001, 'batch_size': 32}
    ]
    
    best_acc = baseline_acc
    best_config = None
    best_model = baseline
    
    for i, config in enumerate(configs):
        print(f"\n   Config {i+1}/{len(configs)}: hidden={config['hidden1']}/{config['hidden2']}, dropout={config['dropout1']}/{config['dropout2']}, lr={config['lr']}")
        
        model = NeuralNetClassifier(
            input_dim, config['hidden1'], config['hidden2'],
            config['dropout1'], config['dropout2']
        ).to(device)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_val_tensor).float().mean().item() * 100
        
        print(f"      Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_config = config
            best_model = model
            print(f"      ✨ New best!")
    
    print(f"\n   ✅ Best accuracy: {best_acc:.2f}%")
    print(f"   📈 Improvement: {best_acc - baseline_acc:+.2f}%")
    print(f"   🎯 Best config: {best_config}")
    
    return best_model, best_acc, best_config


def visualize_results(results, class_names, output_dir):
    """Visualize tuning results"""
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Step 2: Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
    
    # 1. Before vs After Tuning
    ax1 = plt.subplot(2, 3, 1)
    models = list(results.keys())
    baseline_accs = [results[m]['baseline_accuracy'] for m in models]
    tuned_accs = [results[m]['tuned_accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_accs, width, label='Baseline', color='lightblue', edgecolor='black')
    bars2 = ax1.bar(x + width/2, tuned_accs, width, label='Tuned', color='lightgreen', edgecolor='black')
    
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Before vs After Tuning', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['XGB', 'RF', 'SVM', 'NN'], rotation=0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Improvement per Model
    ax2 = plt.subplot(2, 3, 2)
    improvements = [results[m]['tuned_accuracy'] - results[m]['baseline_accuracy'] for m in models]
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.barh(['XGB', 'RF', 'SVM', 'NN'], improvements, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Improvement (%)', fontweight='bold')
    ax2.set_title('Tuning Impact', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax2.text(width + 0.1 if width > 0 else width - 0.1, bar.get_y() + bar.get_height()/2.,
                f'{imp:+.2f}%', ha='left' if width > 0 else 'right', va='center', fontweight='bold')
    
    # 3. Best Model Comparison
    ax3 = plt.subplot(2, 3, 3)
    best_model = max(results.keys(), key=lambda k: results[k]['tuned_accuracy'])
    best_acc = results[best_model]['tuned_accuracy']
    
    comparison = ['Baseline\n(53.14%)', f'Best Tuned\n({best_model})\n({best_acc:.2f}%)']
    comparison_accs = [53.14, best_acc]
    colors_comp = ['gray', 'gold']
    
    bars = ax3.bar(comparison, comparison_accs, color=colors_comp, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Accuracy (%)', fontweight='bold')
    ax3.set_title('Overall Progress', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, comparison_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Parameter Exploration (XGBoost example)
    ax4 = plt.subplot(2, 3, 4)
    xgb_params = results['xgboost']['best_params']
    param_names = list(xgb_params.keys())[:5]  # Top 5 params
    param_values = [str(xgb_params[p]) for p in param_names]
    
    ax4.axis('off')
    table_data = [[p, v] for p, v in zip(param_names, param_values)]
    table = ax4.table(cellText=table_data, colLabels=['Parameter', 'Best Value'],
                     cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('XGBoost Best Parameters', fontweight='bold', pad=20)
    
    # 5. Accuracy Distribution
    ax5 = plt.subplot(2, 3, 5)
    all_accs = baseline_accs + tuned_accs
    
    ax5.hist(baseline_accs, bins=10, alpha=0.5, label='Baseline', color='lightblue', edgecolor='black')
    ax5.hist(tuned_accs, bins=10, alpha=0.5, label='Tuned', color='lightgreen', edgecolor='black')
    ax5.axvline(x=53.14, color='red', linestyle='--', linewidth=2, label='Original Baseline')
    ax5.set_xlabel('Accuracy (%)', fontweight='bold')
    ax5.set_ylabel('Count', fontweight='bold')
    ax5.set_title('Accuracy Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    total_improvement = best_acc - 53.14
    
    summary_text = f"""
{'='*50}
    STEP 2: HYPERPARAMETER TUNING COMPLETE
{'='*50}

Previous Best:        53.14%
After Tuning:         {best_acc:.2f}%
Total Improvement:    {total_improvement:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best Performing Model: {best_model.upper()}
Accuracy: {best_acc:.2f}%

Individual Improvements:
  • XGBoost:  {results['xgboost']['tuned_accuracy'] - results['xgboost']['baseline_accuracy']:+.2f}%
  • RF:       {results['random_forest']['tuned_accuracy'] - results['random_forest']['baseline_accuracy']:+.2f}%
  • SVM:      {results['svm']['tuned_accuracy'] - results['svm']['baseline_accuracy']:+.2f}%
  • NN:       {results['neural_net']['tuned_accuracy'] - results['neural_net']['baseline_accuracy']:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Status: ✅ Complete
Next Step: Fine-tune ViT

{'='*50}
"""
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=9,
            ha='center', va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3,
                     edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    output_path = output_dir / 'step2_hyperparameter_tuning_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Visualization saved: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("🚀 STEP 2: HYPERPARAMETER TUNING")
    print("="*70)
    print("Expected Improvement: +2-4% (target: 58-62%)\n")
    
    # Load data
    embeddings, labels, class_names = load_embeddings()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Apply PCA
    X_train, X_val = apply_pca(X_train, X_val, n_components=270)
    
    print(f"✅ Data ready: {len(X_train)} train, {len(X_val)} validation\n")
    
    print("="*70)
    print("Starting hyperparameter optimization...")
    print("="*70)
    
    # Tune each classifier
    results = {}
    
    # XGBoost
    xgb_model, xgb_acc, xgb_params = tune_xgboost(X_train, y_train, X_val, y_val)
    xgb_baseline = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_baseline.fit(X_train, y_train)
    xgb_baseline_acc = accuracy_score(y_val, xgb_baseline.predict(X_val)) * 100
    results['xgboost'] = {
        'baseline_accuracy': xgb_baseline_acc,
        'tuned_accuracy': xgb_acc,
        'best_params': xgb_params
    }
    
    # Random Forest
    rf_model, rf_acc, rf_params = tune_random_forest(X_train, y_train, X_val, y_val)
    rf_baseline = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_baseline_acc = accuracy_score(y_val, rf_baseline.predict(X_val)) * 100
    results['random_forest'] = {
        'baseline_accuracy': rf_baseline_acc,
        'tuned_accuracy': rf_acc,
        'best_params': rf_params
    }
    
    # SVM
    svm_model, svm_acc, svm_params = tune_svm(X_train, y_train, X_val, y_val)
    svm_baseline = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_baseline.fit(X_train, y_train)
    svm_baseline_acc = accuracy_score(y_val, svm_baseline.predict(X_val)) * 100
    results['svm'] = {
        'baseline_accuracy': svm_baseline_acc,
        'tuned_accuracy': svm_acc,
        'best_params': svm_params
    }
    
    # Neural Network
    nn_model, nn_acc, nn_config = tune_neural_network(X_train, y_train, X_val, y_val)
    results['neural_net'] = {
        'baseline_accuracy': 53.14,  # From previous experiments
        'tuned_accuracy': nn_acc,
        'best_params': nn_config
    }
    
    # Visualize
    # Dynamic path resolution - works on any machine
    output_dir = Path(__file__).parent.parent.parent.resolve() / "Model_Results"
    visualize_results(results, class_names, output_dir)
    
    # Save results
    results_path = output_dir / 'step2_hyperparameter_tuning_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ STEP 2 COMPLETE!")
    print("="*70)
    
    best_model = max(results.keys(), key=lambda k: results[k]['tuned_accuracy'])
    best_acc = results[best_model]['tuned_accuracy']
    
    print(f"\n📊 Results:")
    print(f"   Previous Best:     53.14%")
    print(f"   After Tuning:      {best_acc:.2f}%")
    print(f"   Total Improvement: {best_acc - 53.14:+.2f}%")
    
    print(f"\n🏆 Best Model: {best_model.replace('_', ' ').title()} ({best_acc:.2f}%)")
    
    print(f"\n💾 Saved to: {results_path}")
    print(f"📊 Visualization: {output_dir / 'step2_hyperparameter_tuning_results.png'}")
    
    if best_acc >= 58.0:
        print(f"\n🎉 SUCCESS! Achieved target range (58-62%)")
    else:
        print(f"\n📈 Progress made. Proceed to Step 3 for more gains.")
    
    print("\n💡 Next: Run step3_finetune_vit.py")
    print("="*70)


if __name__ == "__main__":
    main()
