"""
Train Classifiers on RGB vs Grayscale Embeddings
Final comparison to determine which representation is better
"""

import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)


class NeuralClassifier(nn.Module):
    """Simple neural network classifier"""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def load_embeddings(embedding_path):
    """Load embeddings from .npz file"""
    data = np.load(embedding_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    classes = data['classes'].tolist() if 'classes' in data else ['BUY', 'HOLD', 'SELL']
    return embeddings, labels, classes


def compress_embeddings_pca(X_train, X_val, n_components=270):
    """Compress embeddings using PCA dimensionality reduction"""
    print(f"\n🔄 Compressing embeddings: {X_train.shape[1]} → {n_components} dimensions")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_compressed = pca.fit_transform(X_train)
    X_val_compressed = pca.transform(X_val)
    
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"   ✅ Variance preserved: {explained_variance:.2f}%")
    print(f"   Compressed shape: {X_train_compressed.shape}")
    
    return X_train_compressed, X_val_compressed, pca, explained_variance


def train_sklearn_classifier(X_train, y_train, X_val, y_val, classifier_type, classes):
    """Train scikit-learn classifier with class weights for imbalanced data"""
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Calculate sample weights for XGBoost
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    
    classifiers = {
        'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, 
                                      n_jobs=-1, class_weight='balanced'),
        'xgboost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                      random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        'logistic': LogisticRegression(max_iter=1000, random_state=42, 
                                        n_jobs=-1, class_weight='balanced'),
        'svm': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, class_weight='balanced')
    }
    
    clf = classifiers[classifier_type]
    
    # Train with class weights (XGBoost uses sample_weight instead of class_weight)
    if classifier_type == 'xgboost':
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_train, y_train)
    
    # Evaluate
    val_acc = clf.score(X_val, y_val)
    y_pred = clf.predict(X_val)
    
    return clf, val_acc, y_pred


def train_neural_classifier(X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """Train PyTorch neural network classifier with class weights for imbalanced data"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = NeuralClassifier(input_dim, num_classes=3)
    model.to(device)
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_device = X_val_tensor.to(device)
            outputs = model(X_val_device)
            _, predicted = outputs.max(1)
            val_acc = (predicted.cpu() == y_val_tensor).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor.to(device))
        _, y_pred = outputs.max(1)
        y_pred = y_pred.cpu().numpy()
    
    return model, best_val_acc, y_pred


def evaluate_all_classifiers(embeddings, labels, classes, version_name, compress=False, n_components=270):
    """Train and evaluate all classifiers"""
    print(f"\n{'='*70}")
    print(f"🎯 Training Classifiers on {version_name.upper()} Embeddings")
    print(f"{'='*70}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Display class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    print(f"\n📊 Data split: {len(X_train)} train, {len(X_val)} validation")
    print(f"   Original dimensions: {X_train.shape[1]}")
    print(f"   Class distribution: {dict(zip([classes[i] for i in unique], counts))}")
    print(f"   Using class weights to handle imbalance")
    
    # Optional PCA compression
    variance_explained = 100.0
    if compress:
        X_train, X_val, pca, variance_explained = compress_embeddings_pca(
            X_train, X_val, n_components=n_components
        )
    
    results = {}
    
    # Sklearn classifiers
    sklearn_classifiers = ['xgboost', 'rf', 'logistic', 'svm']
    
    for clf_type in sklearn_classifiers:
        print(f"\n   Training {clf_type.upper()}...", end=' ')
        try:
            clf, val_acc, y_pred = train_sklearn_classifier(
                X_train, y_train, X_val, y_val, clf_type, classes
            )
            print(f"✅ Accuracy: {val_acc*100:.2f}%")
            
            results[clf_type] = {
                'accuracy': val_acc,
                'classification_report': classification_report(y_val, y_pred, target_names=classes, output_dict=True),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Neural network
    print(f"\n   Training Neural Network...", end=' ')
    try:
        model, val_acc, y_pred = train_neural_classifier(
            X_train, y_train, X_val, y_val, epochs=30
        )
        print(f"✅ Accuracy: {val_acc*100:.2f}%")
        
        results['neural_net'] = {
            'accuracy': val_acc,
            'classification_report': classification_report(y_val, y_pred, target_names=classes, output_dict=True),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Add compression metadata
    results['_metadata'] = {
        'compressed': compress,
        'n_components': n_components if compress else X_train.shape[1],
        'variance_explained': float(variance_explained)
    }
    
    return results


def create_final_comparison(rgb_results, gray_results, class_names, output_dir):
    """Create final comparison visualization"""
    print(f"\n📊 Creating final comparison dashboard...")
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('FINAL VERDICT: RGB vs Grayscale Embeddings - Classifier Performance', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Overall Accuracy Comparison
    ax1 = plt.subplot(2, 4, 1)
    classifiers = [k for k in rgb_results.keys() if not k.startswith('_')]
    x = np.arange(len(classifiers))
    width = 0.35
    
    rgb_accs = [rgb_results[clf]['accuracy'] * 100 for clf in classifiers]
    gray_accs = [gray_results[clf]['accuracy'] * 100 for clf in classifiers]
    
    bars1 = ax1.bar(x - width/2, rgb_accs, width, label='RGB', color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, gray_accs, width, label='Grayscale', color='#6C757D', alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Classifier Performance Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.upper()[:6] for c in classifiers], rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # 2. Average Performance
    ax2 = plt.subplot(2, 4, 2)
    avg_rgb = np.mean(rgb_accs)
    avg_gray = np.mean(gray_accs)
    
    bars = ax2.bar(['RGB', 'Grayscale'], [avg_rgb, avg_gray], 
                   color=['#2E86AB', '#6C757D'], edgecolor='black', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Average Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Across All Classifiers', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 3. Best Classifier for Each
    ax3 = plt.subplot(2, 4, 3)
    best_rgb = max(
        ((k, v) for k, v in rgb_results.items() if not k.startswith('_')),
        key=lambda x: x[1]['accuracy']
    )
    best_gray = max(
        ((k, v) for k, v in gray_results.items() if not k.startswith('_')),
        key=lambda x: x[1]['accuracy']
    )
    
    bars = ax3.bar(['RGB\nBest', 'Grayscale\nBest'], 
                   [best_rgb[1]['accuracy'] * 100, best_gray[1]['accuracy'] * 100],
                   color=['#2E86AB', '#6C757D'], edgecolor='black', linewidth=2, alpha=0.8)
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Best Classifier Performance', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, (name, _) in zip(bars, [best_rgb, best_gray]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%\n({name.upper()})', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # 4. Win Count
    ax4 = plt.subplot(2, 4, 4)
    rgb_wins = sum(1 for clf in classifiers if rgb_results[clf]['accuracy'] > gray_results[clf]['accuracy'])
    gray_wins = sum(1 for clf in classifiers if gray_results[clf]['accuracy'] > rgb_results[clf]['accuracy'])
    ties = len(classifiers) - rgb_wins - gray_wins
    
    bars = ax4.bar(['RGB Wins', 'Gray Wins', 'Ties'], [rgb_wins, gray_wins, ties],
                   color=['#2E86AB', '#6C757D', '#F5B700'], edgecolor='black', linewidth=2, alpha=0.8)
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Head-to-Head Results', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 5. Best RGB Confusion Matrix
    ax5 = plt.subplot(2, 4, 5)
    rgb_cm = np.array(best_rgb[1]['confusion_matrix'])
    sns.heatmap(rgb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax5, cbar=False, square=True)
    ax5.set_title(f'RGB Best: {best_rgb[0].upper()}\n{best_rgb[1]["accuracy"]*100:.2f}%', 
                  fontsize=12, fontweight='bold')
    ax5.set_ylabel('True', fontsize=10)
    ax5.set_xlabel('Predicted', fontsize=10)
    
    # 6. Best Grayscale Confusion Matrix
    ax6 = plt.subplot(2, 4, 6)
    gray_cm = np.array(best_gray[1]['confusion_matrix'])
    sns.heatmap(gray_cm, annot=True, fmt='d', cmap='Greys', xticklabels=class_names,
                yticklabels=class_names, ax=ax6, cbar=False, square=True)
    ax6.set_title(f'Grayscale Best: {best_gray[0].upper()}\n{best_gray[1]["accuracy"]*100:.2f}%',
                  fontsize=12, fontweight='bold')
    ax6.set_ylabel('True', fontsize=10)
    ax6.set_xlabel('Predicted', fontsize=10)
    
    # 7. Per-Class F1 Comparison (Best Classifiers)
    ax7 = plt.subplot(2, 4, 7)
    x = np.arange(len(class_names))
    width = 0.35
    
    rgb_f1 = [best_rgb[1]['classification_report'][cls]['f1-score'] * 100 for cls in class_names]
    gray_f1 = [best_gray[1]['classification_report'][cls]['f1-score'] * 100 for cls in class_names]
    
    bars1 = ax7.bar(x - width/2, rgb_f1, width, label='RGB', color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax7.bar(x + width/2, gray_f1, width, label='Grayscale', color='#6C757D', alpha=0.8, edgecolor='black')
    
    ax7.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
    ax7.set_title('Per-Class F1-Score (Best)', fontsize=13, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(class_names)
    ax7.legend(fontsize=9)
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Final Verdict
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Determine winner
    if avg_rgb > avg_gray:
        winner = "RGB"
        winner_color = '#2E86AB'
        margin = avg_rgb - avg_gray
    elif avg_gray > avg_rgb:
        winner = "Grayscale"
        winner_color = '#6C757D'
        margin = avg_gray - avg_rgb
    else:
        winner = "TIE"
        winner_color = '#F5B700'
        margin = 0
    
    verdict_text = f"""
{'='*50}
        🏆 FINAL VERDICT 🏆
{'='*50}

WINNER: {winner}
Margin: {margin:.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Average Performance:
  • RGB:       {avg_rgb:.2f}%
  • Grayscale: {avg_gray:.2f}%

Head-to-Head:
  • RGB wins:       {rgb_wins}/{len(classifiers)}
  • Grayscale wins: {gray_wins}/{len(classifiers)}
  • Ties:           {ties}/{len(classifiers)}

Best Individual:
  • RGB:       {best_rgb[0].upper()} ({best_rgb[1]['accuracy']*100:.2f}%)
  • Grayscale: {best_gray[0].upper()} ({best_gray[1]['accuracy']*100:.2f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDATION:
{'Use RGB embeddings for production!' if winner == 'RGB' else 'Use Grayscale embeddings for production!' if winner == 'Grayscale' else 'Either works - choose simpler!'}

{'='*50}
"""
    
    ax8.text(0.5, 0.5, verdict_text, transform=ax8.transAxes, fontsize=10,
            ha='center', va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.3, 
                     edgecolor='black', linewidth=2))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'FINAL_VERDICT_RGB_vs_Grayscale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved to: {output_path}")
    plt.close()
    
    return {
        'winner': winner,
        'margin': margin,
        'avg_rgb': avg_rgb,
        'avg_gray': avg_gray,
        'rgb_wins': rgb_wins,
        'gray_wins': gray_wins
    }


def main():
    print("="*70)
    print("🚀 CLASSIFIER COMPARISON: RGB vs GRAYSCALE EMBEDDINGS")
    print("="*70)
    
    # Dynamic path resolution - works on any machine
    base_dir = Path(__file__).parent.parent.parent.resolve()
    embeddings_dir = base_dir / "embeddings"
    results_dir = base_dir / "Model_Results"
    
    # Load embeddings
    print("\n📂 Loading embeddings...")
    rgb_embeddings, rgb_labels, class_names = load_embeddings(
        embeddings_dir / 'vit_embeddings_rgb.npz'
    )
    gray_embeddings, gray_labels, _ = load_embeddings(
        embeddings_dir / 'vit_embeddings_grayscale.npz'
    )
    
    print(f"   RGB: {rgb_embeddings.shape}")
    print(f"   Grayscale: {gray_embeddings.shape}")
    
    # Option to compress embeddings
    USE_COMPRESSION = True  # Set to True to compress 768 → 270 dimensions
    N_COMPONENTS = 270
    
    if USE_COMPRESSION:
        print(f"\n⚙️  PCA COMPRESSION ENABLED: {rgb_embeddings.shape[1]} → {N_COMPONENTS} dimensions")
    else:
        print(f"\n⚙️  Using FULL embeddings: {rgb_embeddings.shape[1]} dimensions")
    
    # Train classifiers on both
    rgb_results = evaluate_all_classifiers(
        rgb_embeddings, rgb_labels, class_names, "RGB",
        compress=USE_COMPRESSION, n_components=N_COMPONENTS
    )
    gray_results = evaluate_all_classifiers(
        gray_embeddings, gray_labels, class_names, "Grayscale",
        compress=USE_COMPRESSION, n_components=N_COMPONENTS
    )
    
    # Create final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    verdict = create_final_comparison(rgb_results, gray_results, class_names, results_dir)
    
    # Save results
    final_results = {
        'rgb_classifiers': rgb_results,
        'grayscale_classifiers': gray_results,
        'verdict': verdict
    }
    
    results_path = results_dir / 'classifier_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ CLASSIFIER COMPARISON COMPLETE!")
    print("="*70)
    
    # Show compression info
    if rgb_results.get('_metadata', {}).get('compressed'):
        meta = rgb_results['_metadata']
        print(f"\n📊 COMPRESSION APPLIED:")
        print(f"   Original → Compressed: 768 → {meta['n_components']} dimensions")
        print(f"   Variance preserved: {meta['variance_explained']:.2f}%")
    
    print(f"\n🏆 WINNER: {verdict['winner']}")
    print(f"   Margin: {verdict['margin']:.2f}%")
    print(f"\n   RGB Average: {verdict['avg_rgb']:.2f}%")
    print(f"   Grayscale Average: {verdict['avg_gray']:.2f}%")
    print(f"\n   Head-to-Head:")
    print(f"     RGB wins: {verdict['rgb_wins']}")
    print(f"     Grayscale wins: {verdict['gray_wins']}")
    
    print("\n📂 All output files:")
    print("   • Model_Results/FINAL_VERDICT_RGB_vs_Grayscale.png")
    print("   • Model_Results/classifier_comparison_results.json")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
