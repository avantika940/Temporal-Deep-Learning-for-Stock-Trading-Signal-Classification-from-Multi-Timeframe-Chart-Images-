"""
Step 4: Temporal Context Learning with Sequence Models

This script implements temporal sequence learning for stock signal classification.
It uses LSTM and Transformer models to capture temporal patterns in sequences of
stock chart images.

Features:
- SequenceDataset for handling temporal sequences
- TemporalViTLSTM model combining ViT embeddings with LSTM
- TemporalTransformer model with self-attention
- Training for 15 epochs with early stopping
- Comprehensive evaluation and visualization
- Comparison with non-temporal baselines

Author: Advanced ML Pipeline
Date: February 2026
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
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


class SequenceDataset(Dataset):
    """
    Dataset class for temporal sequences of stock chart images.
    
    Each sample is a sequence of consecutive images with the same label.
    This simulates temporal patterns in stock trading signals.
    
    Args:
        root_dir (str): Root directory containing BUY, SELL, HOLD folders
        sequence_length (int): Number of images in each sequence
        transform (callable, optional): Optional transform for images
        mode (str): 'train' or 'test' mode
    """
    
    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 5,
        transform=None,
        mode: str = 'train'
    ):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.mode = mode
        
        self.classes = ['BUY', 'HOLD', 'SELL']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.sequences = []
        self.labels = []
        
        # Load all images grouped by class
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                continue
            
            # Get all images for this class
            image_paths = []
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
            
            # Create sequences from images
            # For simplicity, we'll create overlapping sequences
            for i in range(len(image_paths) - sequence_length + 1):
                sequence = image_paths[i:i + sequence_length]
                self.sequences.append(sequence)
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"{mode.capitalize()} dataset: {len(self.sequences)} sequences created")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence_paths = self.sequences[idx]
        label = self.labels[idx]
        
        # Load all images in the sequence
        images = []
        for img_path in sequence_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # Stack images into a tensor: [seq_len, C, H, W]
        sequence_tensor = torch.stack(images)
        
        return sequence_tensor, label


class TemporalViTLSTM(nn.Module):
    """
    Temporal model combining Vision Transformer with LSTM.
    
    Architecture:
    - ViT extracts features from each image in the sequence
    - LSTM processes the temporal sequence of features
    - Fully connected layers for classification
    
    Args:
        num_classes (int): Number of output classes
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(TemporalViTLSTM, self).__init__()
        
        # Load pre-trained ViT as feature extractor
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_dim = 768  # ViT-Base output dimension
        
        # Freeze ViT parameters (use as fixed feature extractor)
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.vit_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, C, H, W]
            
        Returns:
            Output logits of shape [batch, num_classes]
        """
        batch_size, seq_len, C, H, W = x.shape
        
        # Reshape to process all images through ViT
        x = x.view(batch_size * seq_len, C, H, W)
        
        # Extract features with ViT
        with torch.no_grad():
            vit_outputs = self.vit(x)
            features = vit_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Reshape back to sequences
        features = features.view(batch_size, seq_len, self.vit_dim)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]
        
        # Classify
        logits = self.classifier(last_output)
        
        return logits


class TemporalTransformer(nn.Module):
    """
    Temporal model using Transformer encoder for sequence modeling.
    
    Architecture:
    - ViT extracts features from each image in the sequence
    - Positional encoding added to features
    - Transformer encoder processes the sequence
    - Classification head for final prediction
    
    Args:
        num_classes (int): Number of output classes
        d_model (int): Transformer model dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3
    ):
        super(TemporalTransformer, self).__init__()
        
        # Load pre-trained ViT as feature extractor
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_dim = 768
        
        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Project ViT features to d_model
        self.feature_projection = nn.Linear(self.vit_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, C, H, W]
            
        Returns:
            Output logits of shape [batch, num_classes]
        """
        batch_size, seq_len, C, H, W = x.shape
        
        # Reshape to process all images through ViT
        x = x.view(batch_size * seq_len, C, H, W)
        
        # Extract features with ViT
        with torch.no_grad():
            vit_outputs = self.vit(x)
            features = vit_outputs.last_hidden_state[:, 0, :]
        
        # Reshape back to sequences
        features = features.view(batch_size, seq_len, self.vit_dim)
        
        # Project to d_model
        features = self.feature_projection(features)
        
        # Add positional encoding
        features = self.positional_encoding(features)
        
        # Process through transformer
        transformer_out = self.transformer_encoder(features)
        
        # Use mean pooling over sequence
        pooled = transformer_out.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def get_transforms() -> transforms.Compose:
    """
    Get image transforms for preprocessing.
    
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
    sequence_length: int = 5,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for sequences.
    
    Args:
        sequence_length: Number of images per sequence
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    transform = get_transforms()
    
    # Create full dataset
    full_dataset = SequenceDataset(
        DATA_DIR,
        sequence_length=sequence_length,
        transform=transform,
        mode='full'
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
    num_epochs: int = 15,
    learning_rate: float = 1e-4,
    model_name: str = "Temporal Model"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a temporal model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        model_name: Name of the model for logging
        
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 5
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for sequences, labels in train_pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
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
            for sequences, labels in val_pbar:
                sequences, labels = sequences.to(device), labels.to(device)
                
                outputs = model(sequences)
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
        scheduler.step(val_accuracy)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Check for improvement
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
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
        for sequences, labels in tqdm(data_loader, desc="Evaluating"):
            sequences = sequences.to(device)
            
            outputs = model(sequences)
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
    lstm_history: Dict[str, List[float]],
    transformer_history: Dict[str, List[float]],
    lstm_results: Dict[str, Any],
    transformer_results: Dict[str, Any],
    save_dir: str
):
    """
    Create comprehensive visualizations of temporal model results.
    
    Args:
        lstm_history: Training history for LSTM model
        transformer_history: Training history for Transformer model
        lstm_results: Evaluation results for LSTM model
        transformer_results: Evaluation results for Transformer model
        save_dir: Directory to save visualizations
    """
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. LSTM Training Loss
    ax1 = plt.subplot(2, 3, 1)
    epochs_lstm = range(1, len(lstm_history['train_loss']) + 1)
    ax1.plot(epochs_lstm, lstm_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_lstm, lstm_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('ViT-LSTM: Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. LSTM Training Accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs_lstm, lstm_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs_lstm, lstm_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('ViT-LSTM: Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Transformer Training Loss
    ax3 = plt.subplot(2, 3, 3)
    epochs_trans = range(1, len(transformer_history['train_loss']) + 1)
    ax3.plot(epochs_trans, transformer_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax3.plot(epochs_trans, transformer_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Temporal Transformer: Training and Validation Loss', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Transformer Training Accuracy
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs_trans, transformer_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax4.plot(epochs_trans, transformer_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Temporal Transformer: Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Model Comparison
    ax5 = plt.subplot(2, 3, 5)
    models = ['ViT-LSTM', 'Temporal Transformer']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [lstm_results[metric], transformer_results[metric]]
        ax5.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax5.set_xlabel('Model', fontsize=12)
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(x + width * 1.5)
    ax5.set_xticklabels(models)
    ax5.legend(fontsize=10)
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Confusion Matrices
    ax6 = plt.subplot(2, 3, 6)
    
    # Determine which model is better
    if lstm_results['accuracy'] > transformer_results['accuracy']:
        best_model = 'ViT-LSTM'
        best_cm = np.array(lstm_results['confusion_matrix'])
    else:
        best_model = 'Temporal Transformer'
        best_cm = np.array(transformer_results['confusion_matrix'])
    
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=['BUY', 'HOLD', 'SELL'],
                yticklabels=['BUY', 'HOLD', 'SELL'])
    ax6.set_xlabel('Predicted', fontsize=12)
    ax6.set_ylabel('Actual', fontsize=12)
    ax6.set_title(f'Confusion Matrix - {best_model}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'step4_temporal_context_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main():
    """
    Main execution function for Step 4: Temporal Context Learning.
    """
    print("\n" + "="*80)
    print("STEP 4: TEMPORAL CONTEXT LEARNING WITH SEQUENCE MODELS")
    print("="*80)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Device: {device}")
    
    start_time = time.time()
    
    # Parameters
    SEQUENCE_LENGTH = 5
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    
    # 1. Create data loaders
    print("\n[1/5] Creating sequence data loaders...")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    train_loader, val_loader = create_data_loaders(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    # 2. Train ViT-LSTM model
    print("\n[2/5] Training ViT-LSTM model...")
    lstm_model = TemporalViTLSTM(
        num_classes=3,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    lstm_model, lstm_history = train_model(
        lstm_model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=1e-4,
        model_name="ViT-LSTM"
    )
    
    # 3. Train Temporal Transformer model
    print("\n[3/5] Training Temporal Transformer model...")
    transformer_model = TemporalTransformer(
        num_classes=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.3
    )
    transformer_model, transformer_history = train_model(
        transformer_model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=1e-4,
        model_name="Temporal Transformer"
    )
    
    # 4. Evaluate both models
    print("\n[4/5] Evaluating models...")
    lstm_results = evaluate_model(lstm_model, val_loader, "ViT-LSTM")
    transformer_results = evaluate_model(transformer_model, val_loader, "Temporal Transformer")
    
    # 5. Visualize results
    print("\n[5/5] Creating visualizations...")
    visualize_results(
        lstm_history,
        transformer_history,
        lstm_results,
        transformer_results,
        RESULTS_DIR
    )
    
    # Save results to JSON
    results = {
        'step': 'step4_temporal_context',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'sequence_length': SEQUENCE_LENGTH,
        'num_epochs': NUM_EPOCHS,
        'lstm_model': {
            'name': 'ViT-LSTM',
            'final_train_accuracy': lstm_history['train_acc'][-1],
            'final_val_accuracy': lstm_history['val_acc'][-1],
            'best_val_accuracy': max(lstm_history['val_acc']),
            'evaluation_results': lstm_results
        },
        'transformer_model': {
            'name': 'Temporal Transformer',
            'final_train_accuracy': transformer_history['train_acc'][-1],
            'final_val_accuracy': transformer_history['val_acc'][-1],
            'best_val_accuracy': max(transformer_history['val_acc']),
            'evaluation_results': transformer_results
        },
        'total_time_seconds': time.time() - start_time
    }
    
    results_path = os.path.join(RESULTS_DIR, 'step4_temporal_context_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*80)
    print("STEP 4 COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary
    print("\n📊 SUMMARY:")
    print(f"  ⏱️  Total Time: {time.time() - start_time:.2f} seconds")
    print(f"\n  🔷 ViT-LSTM Model:")
    print(f"     Best Val Accuracy: {max(lstm_history['val_acc']):.2f}%")
    print(f"     Test Accuracy: {lstm_results['accuracy']:.4f}")
    print(f"     F1-Score: {lstm_results['f1_score']:.4f}")
    print(f"\n  🔶 Temporal Transformer Model:")
    print(f"     Best Val Accuracy: {max(transformer_history['val_acc']):.2f}%")
    print(f"     Test Accuracy: {transformer_results['accuracy']:.4f}")
    print(f"     F1-Score: {transformer_results['f1_score']:.4f}")
    
    # Determine best model
    if lstm_results['accuracy'] > transformer_results['accuracy']:
        print(f"\n  🏆 Best Model: ViT-LSTM (Accuracy: {lstm_results['accuracy']:.4f})")
    else:
        print(f"\n  🏆 Best Model: Temporal Transformer (Accuracy: {transformer_results['accuracy']:.4f})")
    
    print(f"\n  💾 Results saved to: {results_path}")
    print(f"  📊 Visualization saved to: {os.path.join(RESULTS_DIR, 'step4_temporal_context_results.png')}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("  ➡️  Run step5_multimodal_features.py to add multi-modal feature fusion")
    print("      This will combine visual features with technical indicators")
    print("="*80)


if __name__ == "__main__":
    main()
