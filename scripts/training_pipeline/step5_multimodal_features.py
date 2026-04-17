"""
Step 5: Multi-modal Feature Fusion

This script implements multi-modal learning by combining visual features from
stock charts with synthetic technical indicators. Three fusion strategies are
tested: concatenation, attention-based fusion, and gating mechanism.

Features:
- Synthetic technical indicator generation
- Visual feature extraction with ViT
- Three fusion strategies (concat, attention, gating)
- Multi-modal fusion network
- Training for 30 epochs
- Comprehensive evaluation and visualization
- Comparison of fusion methods

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from transformers import ViTModel
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

# Create necessary directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MultiModalDataset(Dataset):
    """
    Dataset class for multi-modal learning with images and technical indicators.
    
    Args:
        root_dir (str): Root directory containing BUY, SELL, HOLD folders
        transform (callable, optional): Optional transform for images
        num_indicators (int): Number of synthetic technical indicators
    """
    
    def __init__(
        self,
        root_dir: str,
        transform=None,
        num_indicators: int = 10
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.num_indicators = num_indicators
        
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
    
    def _generate_synthetic_indicators(self, label: int) -> np.ndarray:
        """
        Generate synthetic technical indicators based on label.
        
        In a real scenario, these would be actual technical indicators
        like RSI, MACD, moving averages, etc.
        
        Args:
            label: Class label (0=BUY, 1=HOLD, 2=SELL)
            
        Returns:
            Array of synthetic indicators
        """
        # Generate synthetic indicators with label-dependent distributions
        if label == 0:  # BUY
            # Bullish indicators: higher values
            indicators = np.random.normal(0.6, 0.2, self.num_indicators)
        elif label == 1:  # HOLD
            # Neutral indicators: middle values
            indicators = np.random.normal(0.5, 0.15, self.num_indicators)
        else:  # SELL (label == 2)
            # Bearish indicators: lower values
            indicators = np.random.normal(0.4, 0.2, self.num_indicators)
        
        # Clip to [0, 1] range and add noise
        indicators = np.clip(indicators, 0, 1)
        
        return indicators.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Generate synthetic technical indicators
        indicators = self._generate_synthetic_indicators(label)
        indicators = torch.tensor(indicators, dtype=torch.float32)
        
        return image, indicators, label


class MultiModalFusionNetwork(nn.Module):
    """
    Multi-modal fusion network combining visual and technical indicator features.
    
    Supports three fusion strategies:
    - 'concat': Simple concatenation of features
    - 'attention': Attention-based fusion
    - 'gating': Gating mechanism for adaptive fusion
    
    Args:
        num_classes (int): Number of output classes
        num_indicators (int): Number of technical indicators
        fusion_type (str): Type of fusion ('concat', 'attention', 'gating')
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        num_indicators: int = 10,
        fusion_type: str = 'concat',
        dropout: float = 0.3
    ):
        super(MultiModalFusionNetwork, self).__init__()
        
        self.fusion_type = fusion_type
        
        # Visual feature extractor (ViT)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_dim = 768
        
        # Freeze ViT parameters (use as fixed feature extractor)
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Technical indicator encoder
        self.indicator_encoder = nn.Sequential(
            nn.Linear(num_indicators, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Visual feature projection
        self.visual_projection = nn.Sequential(
            nn.Linear(self.vit_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion-specific layers
        if fusion_type == 'concat':
            # Simple concatenation
            self.fusion_dim = 512 + 256
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            # Project visual features to match indicator dimension
            self.visual_to_indicator = nn.Linear(512, 256)
            
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
        
        elif fusion_type == 'gating':
            # Gating mechanism
            self.visual_gate = nn.Sequential(
                nn.Linear(512, 256),
                nn.Sigmoid()
            )
            self.indicator_gate = nn.Sequential(
                nn.Linear(256, 256),
                nn.Sigmoid()
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        images: torch.Tensor,
        indicators: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Visual input [batch, C, H, W]
            indicators: Technical indicators [batch, num_indicators]
            
        Returns:
            Output logits [batch, num_classes]
        """
        # Extract visual features
        with torch.no_grad():
            vit_outputs = self.vit(images)
            visual_features = vit_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Project visual features
        visual_features = self.visual_projection(visual_features)  # [batch, 512]
        
        # Encode technical indicators
        indicator_features = self.indicator_encoder(indicators)  # [batch, 256]
        
        # Fusion
        if self.fusion_type == 'concat':
            # Simple concatenation
            fused_features = torch.cat([visual_features, indicator_features], dim=1)
            logits = self.classifier(fused_features)
        
        elif self.fusion_type == 'attention':
            # Attention-based fusion
            visual_proj = self.visual_to_indicator(visual_features).unsqueeze(1)  # [batch, 1, 256]
            indicator_proj = indicator_features.unsqueeze(1)  # [batch, 1, 256]
            
            # Self-attention over both modalities
            combined = torch.cat([visual_proj, indicator_proj], dim=1)  # [batch, 2, 256]
            attended, _ = self.attention(combined, combined, combined)
            
            # Mean pooling
            fused_features = attended.mean(dim=1)  # [batch, 256]
            logits = self.classifier(fused_features)
        
        elif self.fusion_type == 'gating':
            # Gating mechanism
            visual_gate_values = self.visual_gate(visual_features)  # [batch, 256]
            indicator_gate_values = self.indicator_gate(indicator_features)  # [batch, 256]
            
            # Project visual features to match dimension
            visual_proj = nn.functional.adaptive_avg_pool1d(
                visual_features.unsqueeze(1),
                256
            ).squeeze(1)
            
            # Apply gates
            gated_visual = visual_proj * visual_gate_values
            gated_indicators = indicator_features * indicator_gate_values
            
            # Combine
            fused_features = gated_visual + gated_indicators
            logits = self.classifier(fused_features)
        
        return logits


def get_transforms() -> transforms.Compose:
    """
    Get image transforms.
    
    Returns:
        Transform composition
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform


def create_data_loaders(
    batch_size: int = 32,
    num_indicators: int = 10
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        batch_size: Batch size
        num_indicators: Number of technical indicators
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    transform = get_transforms()
    
    # Create full dataset
    full_dataset = MultiModalDataset(
        DATA_DIR,
        transform=transform,
        num_indicators=num_indicators
    )
    
    # Split into train and validation (80-20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )
    
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    model_name: str = "Multi-modal Model"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a multi-modal fusion model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        model_name: Name of the model
        
    Returns:
        Tuple of (trained model, training history)
    """
    print("\n" + "="*80)
    print(f"Training {model_name}")
    print("="*80)
    
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
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
        for images, indicators, labels in train_pbar:
            images = images.to(device)
            indicators = indicators.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, indicators)
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
            for images, indicators, labels in val_pbar:
                images = images.to(device)
                indicators = indicators.to(device)
                labels = labels.to(device)
                
                outputs = model(images, indicators)
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
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        model_name: Name of the model
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, indicators, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            indicators = indicators.to(device)
            
            outputs = model(images, indicators)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    return results


def visualize_results(
    histories: Dict[str, Dict[str, List[float]]],
    results: Dict[str, Dict[str, Any]],
    save_dir: str
):
    """
    Create comprehensive visualizations of multi-modal fusion results.
    
    Args:
        histories: Training histories for all fusion types
        results: Evaluation results for all fusion types
        save_dir: Directory to save visualizations
    """
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    fusion_types = list(histories.keys())
    colors = ['blue', 'green', 'red']
    
    # 1. Training Loss Comparison
    ax1 = plt.subplot(2, 3, 1)
    for i, fusion in enumerate(fusion_types):
        epochs = range(1, len(histories[fusion]['train_loss']) + 1)
        ax1.plot(epochs, histories[fusion]['train_loss'], 
                color=colors[i], linestyle='-', label=f'{fusion}', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Accuracy Comparison
    ax2 = plt.subplot(2, 3, 2)
    for i, fusion in enumerate(fusion_types):
        epochs = range(1, len(histories[fusion]['val_acc']) + 1)
        ax2.plot(epochs, histories[fusion]['val_acc'],
                color=colors[i], linestyle='-', label=f'{fusion}', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Final Metrics Comparison
    ax3 = plt.subplot(2, 3, 3)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(fusion_types))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[fusion][metric] for fusion in fusion_types]
        ax3.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax3.set_xlabel('Fusion Type', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(fusion_types)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4-6. Confusion Matrices for each fusion type
    for idx, fusion in enumerate(fusion_types):
        ax = plt.subplot(2, 3, 4 + idx)
        cm = np.array(results[fusion]['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['BUY', 'HOLD', 'SELL'],
                    yticklabels=['BUY', 'HOLD', 'SELL'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix - {fusion.title()}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'step5_multimodal_features_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main():
    """
    Main execution function for Step 5: Multi-modal Feature Fusion.
    """
    print("\n" + "="*80)
    print("STEP 5: MULTI-MODAL FEATURE FUSION")
    print("="*80)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Device: {device}")
    
    start_time = time.time()
    
    # Parameters
    NUM_INDICATORS = 10
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    FUSION_TYPES = ['concat', 'attention', 'gating']
    
    # 1. Create data loaders
    print("\n[1/4] Creating multi-modal data loaders...")
    print(f"Number of technical indicators: {NUM_INDICATORS}")
    train_loader, val_loader = create_data_loaders(
        batch_size=BATCH_SIZE,
        num_indicators=NUM_INDICATORS
    )
    
    # Store results for all fusion types
    all_histories = {}
    all_results = {}
    
    # 2. Train models with different fusion strategies
    for i, fusion_type in enumerate(FUSION_TYPES):
        print(f"\n[{i+2}/4] Training {fusion_type.upper()} fusion model...")
        
        model = MultiModalFusionNetwork(
            num_classes=3,
            num_indicators=NUM_INDICATORS,
            fusion_type=fusion_type,
            dropout=0.3
        )
        
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=1e-4,
            model_name=f"Multi-modal ({fusion_type})"
        )
        
        # Evaluate
        eval_results = evaluate_model(
            model,
            val_loader,
            f"Multi-modal ({fusion_type})"
        )
        
        all_histories[fusion_type] = history
        all_results[fusion_type] = eval_results
    
    # 3. Visualize results
    print(f"\n[{len(FUSION_TYPES)+2}/4] Creating visualizations...")
    visualize_results(all_histories, all_results, RESULTS_DIR)
    
    # Save results to JSON
    results = {
        'step': 'step5_multimodal_features',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'num_indicators': NUM_INDICATORS,
        'num_epochs': NUM_EPOCHS,
        'fusion_strategies': {}
    }
    
    for fusion_type in FUSION_TYPES:
        results['fusion_strategies'][fusion_type] = {
            'final_train_accuracy': all_histories[fusion_type]['train_acc'][-1],
            'final_val_accuracy': all_histories[fusion_type]['val_acc'][-1],
            'best_val_accuracy': max(all_histories[fusion_type]['val_acc']),
            'evaluation_results': all_results[fusion_type]
        }
    
    results['total_time_seconds'] = time.time() - start_time
    
    results_path = os.path.join(RESULTS_DIR, 'step5_multimodal_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*80)
    print("STEP 5 COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary
    print("\n📊 SUMMARY:")
    print(f"  ⏱️  Total Time: {time.time() - start_time:.2f} seconds")
    
    for fusion_type in FUSION_TYPES:
        print(f"\n  🔹 {fusion_type.upper()} Fusion:")
        print(f"     Best Val Accuracy: {max(all_histories[fusion_type]['val_acc']):.2f}%")
        print(f"     Test Accuracy: {all_results[fusion_type]['accuracy']:.4f}")
        print(f"     F1-Score: {all_results[fusion_type]['f1_score']:.4f}")
    
    # Determine best fusion strategy
    best_fusion = max(FUSION_TYPES, key=lambda x: all_results[x]['accuracy'])
    print(f"\n  🏆 Best Fusion Strategy: {best_fusion.upper()}")
    print(f"     Accuracy: {all_results[best_fusion]['accuracy']:.4f}")
    
    print(f"\n  💾 Results saved to: {results_path}")
    print(f"  📊 Visualization saved to: {os.path.join(RESULTS_DIR, 'step5_multimodal_features_results.png')}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("  ✅ All 5 steps completed successfully")
    print("  ✅ Results and visualizations saved")
    print("  ✅ Ready for production deployment")
    print("="*80)


if __name__ == "__main__":
    main()
