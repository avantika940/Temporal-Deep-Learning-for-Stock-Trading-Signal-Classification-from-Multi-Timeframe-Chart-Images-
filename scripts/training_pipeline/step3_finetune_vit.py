"""
Step 3: Fine-tune Vision Transformer (ViT) for Stock Signal Classification

This script fine-tunes a pre-trained Vision Transformer model on the stock signal
classification task. The fine-tuned model is then used to extract embeddings,
which are used to train classifiers for improved performance.

Features:
- Fine-tunes ViT-Base model for 20 epochs
- Extracts embeddings from fine-tuned model
- Applies PCA for dimensionality reduction
- Trains multiple classifiers on extracted embeddings
- Comprehensive visualization of results
- Saves all results to Model_Results/

Author: Advanced ML Pipeline
Date: February 2026
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from transformers import ViTForImageClassification, ViTModel, ViTConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dynamic path resolution - works on any machine
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = BASE_DIR / "data" / "Adani_MTF_Images_224x224"
RESULTS_DIR = BASE_DIR / "Model_Results"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class StockImageDataset(Dataset):
    """
    Custom Dataset class for loading stock signal images.
    
    Args:
        root_dir (str): Root directory containing BUY, SELL, HOLD folders
        transform (callable, optional): Optional transform to be applied on images
        
    Attributes:
        classes (list): List of class names ['BUY', 'HOLD', 'SELL']
        class_to_idx (dict): Mapping from class names to indices
    """
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['BUY', 'HOLD', 'SELL']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all images and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data augmentation transforms for training and validation.
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


def create_data_loaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders with 80-20 split.
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform, val_transform = get_transforms()
    
    # Load full dataset
    full_dataset = StockImageDataset(DATA_DIR, transform=train_transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}")
    
    return train_loader, val_loader


def finetune_vit_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 2e-5
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Fine-tune Vision Transformer model for stock signal classification.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (fine-tuned model, training history)
    """
    print("\n" + "="*80)
    print("Fine-tuning Vision Transformer Model")
    print("="*80)
    
    # Load pre-trained ViT model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


def extract_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    embedding_type: str = 'cls_token'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from fine-tuned ViT model.
    
    Args:
        model: Fine-tuned ViT model
        data_loader: Data loader for images
        embedding_type: Type of embedding ('cls_token' or 'mean_pooling')
        
    Returns:
        Tuple of (embeddings, labels)
    """
    print("\n" + "="*80)
    print(f"Extracting Embeddings ({embedding_type})")
    print("="*80)
    
    # Get the base ViT model (without classification head)
    base_model = model.vit
    base_model.eval()
    
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            
            # Get hidden states
            outputs = base_model(images)
            hidden_states = outputs.last_hidden_state  # Shape: [batch, seq_len, hidden_dim]
            
            if embedding_type == 'cls_token':
                # Use CLS token (first token)
                batch_embeddings = hidden_states[:, 0, :].cpu().numpy()
            else:  # mean_pooling
                # Average over all tokens
                batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
            
            embeddings_list.append(batch_embeddings)
            labels_list.append(labels.numpy())
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return embeddings, labels


def apply_pca(
    embeddings: np.ndarray,
    n_components: int = 128
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        embeddings: Input embeddings
        n_components: Number of PCA components
        
    Returns:
        Tuple of (reduced embeddings, fitted PCA object)
    """
    print(f"\nApplying PCA (n_components={n_components})...")
    
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance: {explained_variance:.2f}%")
    
    return embeddings_pca, pca


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple classifiers on extracted embeddings.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing results for each classifier
    """
    print("\n" + "="*80)
    print("Training Classifiers on Fine-tuned Embeddings")
    print("="*80)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='rbf',
            gamma='scale',
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    results = {}
    
    # Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        print(f"\n{clf_name}:")
        
        # Train
        start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Training time: {train_time:.2f}s")
        
        results[clf_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': float(train_time),
            'confusion_matrix': cm.tolist()
        }
    
    return results


def visualize_results(
    history: Dict[str, List[float]],
    classifier_results: Dict[str, Dict[str, Any]],
    save_dir: str
):
    """
    Create comprehensive visualizations of results.
    
    Args:
        history: Training history from fine-tuning
        classifier_results: Results from classifier training
        save_dir: Directory to save visualizations
    """
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training and Validation Loss
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Training and Validation Accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Classifier Comparison - Accuracy
    ax3 = plt.subplot(2, 3, 3)
    clf_names = list(classifier_results.keys())
    accuracies = [classifier_results[clf]['accuracy'] for clf in clf_names]
    bars = ax3.bar(range(len(clf_names)), accuracies, color=sns.color_palette("husl", len(clf_names)))
    ax3.set_xticks(range(len(clf_names)))
    ax3.set_xticklabels(clf_names, rotation=45, ha='right')
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Classifier Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{accuracies[i]:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Classifier Comparison - All Metrics
    ax4 = plt.subplot(2, 3, 4)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(clf_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [classifier_results[clf][metric] for clf in clf_names]
        ax4.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax4.set_xlabel('Classifier', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Classifier Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(clf_names, rotation=45, ha='right')
    ax4.legend(fontsize=10)
    ax4.set_ylim([0, 1])
    
    # 5. Confusion Matrix (for best classifier)
    best_clf = max(classifier_results.keys(), key=lambda x: classifier_results[x]['accuracy'])
    cm = np.array(classifier_results[best_clf]['confusion_matrix'])
    
    ax5 = plt.subplot(2, 3, 5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                xticklabels=['BUY', 'HOLD', 'SELL'],
                yticklabels=['BUY', 'HOLD', 'SELL'])
    ax5.set_xlabel('Predicted', fontsize=12)
    ax5.set_ylabel('Actual', fontsize=12)
    ax5.set_title(f'Confusion Matrix - {best_clf}', fontsize=14, fontweight='bold')
    
    # 6. Training Time Comparison
    ax6 = plt.subplot(2, 3, 6)
    train_times = [classifier_results[clf]['training_time'] for clf in clf_names]
    bars = ax6.bar(range(len(clf_names)), train_times, color=sns.color_palette("muted", len(clf_names)))
    ax6.set_xticks(range(len(clf_names)))
    ax6.set_xticklabels(clf_names, rotation=45, ha='right')
    ax6.set_ylabel('Time (seconds)', fontsize=12)
    ax6.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{train_times[i]:.2f}s',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'step3_finetune_vit_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main():
    """
    Main execution function for Step 3: Fine-tune ViT.
    """
    print("\n" + "="*80)
    print("STEP 3: FINE-TUNE VISION TRANSFORMER")
    print("="*80)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Device: {device}")
    
    start_time = time.time()
    
    # 1. Create data loaders
    print("\n[1/6] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(batch_size=32)
    
    # 2. Fine-tune ViT model
    print("\n[2/6] Fine-tuning Vision Transformer...")
    model, history = finetune_vit_model(
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=2e-5
    )
    
    # 3. Extract embeddings
    print("\n[3/6] Extracting embeddings from fine-tuned model...")
    # Combine train and val for embedding extraction
    full_dataset = StockImageDataset(DATA_DIR, transform=get_transforms()[1])
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    embeddings, labels = extract_embeddings(model, full_loader, embedding_type='cls_token')
    
    # Save embeddings
    embeddings_path = os.path.join(EMBEDDINGS_DIR, 'vit_embeddings_finetuned.npz')
    np.savez(embeddings_path, embeddings=embeddings, labels=labels)
    print(f"Embeddings saved to: {embeddings_path}")
    
    # 4. Apply PCA
    print("\n[4/6] Applying PCA for dimensionality reduction...")
    embeddings_pca, pca_model = apply_pca(embeddings, n_components=128)
    
    # 5. Train classifiers
    print("\n[5/6] Training classifiers...")
    # Split embeddings into train and test
    train_size = int(0.8 * len(embeddings_pca))
    indices = np.random.permutation(len(embeddings_pca))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train = embeddings_pca[train_idx]
    y_train = labels[train_idx]
    X_test = embeddings_pca[test_idx]
    y_test = labels[test_idx]
    
    classifier_results = train_classifiers(X_train, y_train, X_test, y_test)
    
    # 6. Visualize results
    print("\n[6/6] Creating visualizations...")
    visualize_results(history, classifier_results, RESULTS_DIR)
    
    # Save results to JSON
    results = {
        'step': 'step3_finetune_vit',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'training_epochs': 20,
        'final_train_accuracy': history['train_acc'][-1],
        'final_val_accuracy': history['val_acc'][-1],
        'best_val_accuracy': max(history['val_acc']),
        'embedding_shape': embeddings.shape,
        'pca_components': 128,
        'pca_explained_variance': float(np.sum(pca_model.explained_variance_ratio_)),
        'classifier_results': classifier_results,
        'total_time_seconds': time.time() - start_time
    }
    
    results_path = os.path.join(RESULTS_DIR, 'step3_finetune_vit_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*80)
    print("STEP 3 COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary
    print("\n📊 SUMMARY:")
    print(f"  ⏱️  Total Time: {time.time() - start_time:.2f} seconds")
    print(f"  🎯 Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"  📈 Best Classifier: {max(classifier_results.keys(), key=lambda x: classifier_results[x]['accuracy'])}")
    print(f"  🎪 Best Classifier Accuracy: {max(classifier_results[clf]['accuracy'] for clf in classifier_results):.4f}")
    print(f"\n  💾 Results saved to: {results_path}")
    print(f"  📊 Visualization saved to: {os.path.join(RESULTS_DIR, 'step3_finetune_vit_results.png')}")
    print(f"  🔢 Embeddings saved to: {embeddings_path}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("  ➡️  Run step4_temporal_context.py to add temporal sequence learning")
    print("      This will leverage LSTM and Transformer models for time-series patterns")
    print("="*80)


if __name__ == "__main__":
    main()
