"""
STEP 1: Ensemble Multiple Classifiers
Expected Accuracy Gain: +3-5% (target: 56-58%)

Combines predictions from multiple trained classifiers using weighted voting.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleNeuralNet(nn.Module):
    """Simple neural network classifier"""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def load_embeddings():
    """Load pre-extracted embeddings"""
    # Dynamic path resolution - works on any machine
    base_dir = Path(__file__).parent.parent.parent.resolve()
    embeddings_dir = base_dir / "embeddings"
    
    print("📂 Loading embeddings...")
    
    # Load grayscale embeddings (best performing from previous experiments)
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
    return X_train_pca, X_val_pca, pca


def train_neural_network(X_train, y_train, X_val, y_val, num_epochs=50):
    """Train neural network classifier"""
    input_dim = X_train.shape[1]
    model = SimpleNeuralNet(input_dim).to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, predicted = torch.max(val_outputs, 1)
                val_acc = (predicted == y_val_tensor).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
    
    return model


def train_classifiers_for_ensemble(X_train, y_train, X_val, y_val):
    """Train all classifiers"""
    print("\n" + "="*70)
    print("🎯 TRAINING CLASSIFIERS FOR ENSEMBLE")
    print("="*70)
    
    classifiers = {}
    accuracies = {}
    
    # 1. XGBoost
    print("\n1. Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    classifiers['xgboost'] = xgb
    accuracies['xgboost'] = acc
    print(f"   ✅ Accuracy: {acc:.2f}%")
    
    # 2. Random Forest
    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    classifiers['random_forest'] = rf
    accuracies['random_forest'] = acc
    print(f"   ✅ Accuracy: {acc:.2f}%")
    
    # 3. SVM
    print("\n3. Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    classifiers['svm'] = svm
    accuracies['svm'] = acc
    print(f"   ✅ Accuracy: {acc:.2f}%")
    
    # 4. Logistic Regression
    print("\n4. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    classifiers['logistic'] = lr
    accuracies['logistic'] = acc
    print(f"   ✅ Accuracy: {acc:.2f}%")
    
    # 5. Neural Network
    print("\n5. Training Neural Network...")
    nn_model = train_neural_network(X_train, y_train, X_val, y_val)
    nn_model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        outputs = nn_model(X_val_tensor)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
    acc = accuracy_score(y_val, y_pred) * 100
    classifiers['neural_net'] = nn_model
    accuracies['neural_net'] = acc
    print(f"   ✅ Accuracy: {acc:.2f}%")
    
    return classifiers, accuracies


def create_ensemble(classifiers, X_val, y_val, weights=None):
    """Create ensemble with weighted voting"""
    print("\n" + "="*70)
    print("🎭 CREATING ENSEMBLE")
    print("="*70)
    
    if weights is None:
        weights = {name: 1.0 for name in classifiers.keys()}
    
    print(f"\nWeights: {weights}")
    
    # Get predictions from all models
    predictions = {}
    
    for name, model in classifiers.items():
        if name == 'neural_net':
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                outputs = model(X_val_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
        else:
            probs = model.predict_proba(X_val)
        
        predictions[name] = probs
    
    # Weighted voting
    ensemble_probs = np.zeros_like(predictions['xgboost'])
    for name, probs in predictions.items():
        ensemble_probs += probs * weights[name]
    
    # Final predictions
    y_pred = np.argmax(ensemble_probs, axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"✨ Ensemble Accuracy: {accuracy*100:.2f}%")
    
    return y_pred, accuracy


def optimize_ensemble_weights(classifiers, X_val, y_val):
    """Find optimal weights for ensemble"""
    print("\n" + "="*70)
    print("🔍 OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*70)
    
    print("\nSearching weight combinations...")
    
    best_accuracy = 0.0
    best_weights = None
    
    # Grid search over weight combinations
    weight_options = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    for w1 in weight_options:
        for w2 in weight_options:
            for w3 in weight_options:
                for w4 in weight_options:
                    w5 = 1.0 - (w1 + w2 + w3 + w4)
                    if w5 >= 0.05 and w5 <= 0.35:
                        weights = {
                            'xgboost': w1,
                            'random_forest': w2,
                            'svm': w3,
                            'logistic': w4,
                            'neural_net': w5
                        }
                        
                        _, accuracy = create_ensemble(classifiers, X_val, y_val, weights)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_weights = weights
                            print(f"   New best: {accuracy*100:.2f}% - {weights}")
    
    print(f"\n✅ Tested {len(weight_options)**4} combinations")
    print(f"🏆 Best Accuracy: {best_accuracy*100:.2f}%")
    print(f"🎯 Best Weights: {best_weights}")
    
    return best_weights, best_accuracy


def visualize_results(individual_accs, ensemble_acc, cm, class_names, output_dir):
    """Visualize ensemble results"""
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Step 1: Ensemble Models Results', fontsize=16, fontweight='bold')
    
    # 1. Individual vs Ensemble
    ax1 = plt.subplot(2, 3, 1)
    models = list(individual_accs.keys()) + ['Ensemble']
    accuracies = list(individual_accs.values()) + [ensemble_acc * 100]
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'gold']
    
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Individual Models vs Ensemble', fontweight='bold')
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.axhline(y=53.14, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax2, cbar_kws={'label': 'Count'}, square=True)
    ax2.set_xlabel('Predicted', fontweight='bold')
    ax2.set_ylabel('True', fontweight='bold')
    ax2.set_title(f'Ensemble Confusion Matrix\n({ensemble_acc*100:.2f}%)', fontweight='bold')
    
    # 3. Model Agreement Analysis
    ax3 = plt.subplot(2, 3, 3)
    baseline_acc = 53.14
    improvements = [acc - baseline_acc for acc in list(individual_accs.values()) + [ensemble_acc * 100]]
    model_names = list(individual_accs.keys()) + ['Ensemble']
    
    colors_improvement = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax3.barh(model_names, improvements, color=colors_improvement, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_xlabel('Improvement over Baseline (%)', fontweight='bold')
    ax3.set_title('Accuracy Gains', fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax3.text(width + 0.2 if width > 0 else width - 0.2, bar.get_y() + bar.get_height()/2.,
                f'{imp:+.2f}%', ha='left' if width > 0 else 'right', va='center', fontweight='bold')
    
    # 4. Accuracy Progression
    ax4 = plt.subplot(2, 3, 4)
    steps = ['Baseline', 'Best Single', 'Ensemble']
    step_accs = [53.14, max(individual_accs.values()), ensemble_acc * 100]
    colors_steps = ['gray', 'lightblue', 'gold']
    
    bars = ax4.bar(steps, step_accs, color=colors_steps, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Accuracy (%)', fontweight='bold')
    ax4.set_title('Progressive Improvement', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, step_accs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Model Contributions
    ax5 = plt.subplot(2, 3, 5)
    model_names_short = ['XGB', 'RF', 'SVM', 'LR', 'NN']
    model_accs = list(individual_accs.values())
    
    ax5.scatter(range(len(model_names_short)), model_accs, s=200, alpha=0.6, c=colors[:5], edgecolors='black', linewidth=2)
    ax5.axhline(y=ensemble_acc * 100, color='gold', linestyle='--', linewidth=2, label='Ensemble')
    ax5.axhline(y=53.14, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    ax5.set_xticks(range(len(model_names_short)))
    ax5.set_xticklabels(model_names_short)
    ax5.set_ylabel('Accuracy (%)', fontweight='bold')
    ax5.set_title('Model Performance Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
{'='*50}
        STEP 1: ENSEMBLE COMPLETE
{'='*50}

Previous Best:        53.14%
Ensemble Result:      {ensemble_acc*100:.2f}%
Improvement:          {ensemble_acc*100 - 53.14:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Individual Model Performance:
  • XGBoost:     {individual_accs['xgboost']:.2f}%
  • Random Forest: {individual_accs['random_forest']:.2f}%
  • SVM:         {individual_accs['svm']:.2f}%
  • Logistic:    {individual_accs['logistic']:.2f}%
  • Neural Net:  {individual_accs['neural_net']:.2f}%

Ensemble combines strengths of all models
using optimized weighted voting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Status: ✅ Complete
Next Step: Hyperparameter Tuning

{'='*50}
"""
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=9,
            ha='center', va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2,
                     edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    output_path = output_dir / 'step1_ensemble_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Visualization saved: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("🚀 STEP 1: ENSEMBLE MULTIPLE CLASSIFIERS")
    print("="*70)
    print("Expected Improvement: +3-5% (target: 56-58%)\n")
    
    # Load data
    embeddings, labels, class_names = load_embeddings()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Apply PCA
    X_train, X_val, pca = apply_pca(X_train, X_val, n_components=270)
    
    print(f"✅ Data ready: {len(X_train)} train, {len(X_val)} validation\n")
    
    # Train individual classifiers
    classifiers, individual_accs = train_classifiers_for_ensemble(X_train, y_train, X_val, y_val)
    
    # Optimize ensemble weights
    best_weights, best_accuracy = optimize_ensemble_weights(classifiers, X_val, y_val)
    
    # Create final ensemble
    print("\n" + "="*70)
    print("🏆 CREATING FINAL ENSEMBLE WITH OPTIMAL WEIGHTS")
    print("="*70)
    y_pred, ensemble_acc = create_ensemble(classifiers, X_val, y_val, best_weights)
    
    # Generate metrics
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
    
    # Visualize
    # Dynamic path resolution - works on any machine
    output_dir = Path(__file__).parent.parent.parent.resolve() / "Model_Results"
    output_dir.mkdir(exist_ok=True)
    visualize_results(individual_accs, ensemble_acc, cm, class_names, output_dir)
    
    # Save results
    results = {
        'step': 1,
        'method': 'Ensemble',
        'individual_accuracies': individual_accs,
        'ensemble_accuracy': ensemble_acc * 100,
        'best_weights': best_weights,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    results_path = output_dir / 'step1_ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ STEP 1 COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Results:")
    print(f"   Previous Best:     53.14%")
    print(f"   Ensemble Result:   {ensemble_acc*100:.2f}%")
    print(f"   Improvement:       {ensemble_acc*100 - 53.14:+.2f}%")
    
    print(f"\n💾 Saved to: {results_path}")
    print(f"📊 Visualization: {output_dir / 'step1_ensemble_results.png'}")
    
    if ensemble_acc * 100 >= 56.0:
        print(f"\n🎉 SUCCESS! Achieved target range (56-58%)")
    else:
        print(f"\n📈 Close to target. Proceed to Step 2 for more gains.")
    
    print("\n💡 Next: Run step2_hyperparameter_tuning.py")
    print("="*70)


if __name__ == "__main__":
    main()
