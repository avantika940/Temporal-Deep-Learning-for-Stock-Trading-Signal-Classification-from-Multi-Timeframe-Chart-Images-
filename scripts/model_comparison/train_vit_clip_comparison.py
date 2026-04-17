"""
Vision Transformer (ViT) vs CLIP Model Comparison
Training and evaluation on Adani MTF stock images (BUY/HOLD/SELL classification)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import json


class StockImageDataset(Dataset):
    """Dataset for stock images with BUY/HOLD/SELL labels"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = ['BUY', 'HOLD', 'SELL']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Load all images
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
        for cls in self.class_names:
            count = self.labels.count(self.class_to_idx[cls])
            print(f"  {cls}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_transforms():
    """Create transforms for training and validation"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_vit_model(num_classes=3):
    """Load pretrained Vision Transformer model"""
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        
        print("Loading Vision Transformer (ViT) model...")
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        
        return model, processor
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return None, None


def load_clip_model(num_classes=3):
    """Load pretrained CLIP model with classification head"""
    try:
        import clip
        
        print("Loading CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        
        # Add classification head on top of CLIP
        class CLIPClassifier(nn.Module):
            def __init__(self, clip_model, num_classes):
                super().__init__()
                self.clip_model = clip_model
                self.classifier = nn.Linear(512, num_classes)  # CLIP ViT-B/32 outputs 512-dim features
                
            def forward(self, images):
                with torch.no_grad():
                    features = self.clip_model.encode_image(images)
                features = features.float()
                return self.classifier(features)
        
        clip_classifier = CLIPClassifier(model, num_classes)
        
        return clip_classifier, preprocess
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return None, None


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, model_name="Model"):
    """Train a model and track performance"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if model_name == "ViT":
                outputs = model(images).logits
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                if model_name == "ViT":
                    outputs = model(images).logits
                else:
                    outputs = model(images)
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        epoch_time = time.time() - start_time
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        print(f'Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'  ✓ Best validation accuracy so far!')
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    return history, best_val_acc


def evaluate_model(model, test_loader, device, class_names, model_name="Model"):
    """Evaluate model and generate detailed metrics"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Evaluating {model_name}'):
            images = images.to(device)
            
            if model_name == "ViT":
                outputs = model(images).logits
            else:
                outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm, all_preds, all_labels


def plot_comparison_results(vit_history, clip_history, vit_cm, clip_cm, class_names, output_dir):
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training Loss Comparison
    ax1 = plt.subplot(2, 4, 1)
    epochs = range(1, len(vit_history['train_loss']) + 1)
    ax1.plot(epochs, vit_history['train_loss'], 'b-o', label='ViT Train', linewidth=2)
    ax1.plot(epochs, clip_history['train_loss'], 'r-s', label='CLIP Train', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss Comparison
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(epochs, vit_history['val_loss'], 'b--o', label='ViT Val', linewidth=2)
    ax2.plot(epochs, clip_history['val_loss'], 'r--s', label='CLIP Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Accuracy Comparison
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(epochs, vit_history['train_acc'], 'b-o', label='ViT Train', linewidth=2)
    ax3.plot(epochs, clip_history['train_acc'], 'r-s', label='CLIP Train', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Training Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Accuracy Comparison
    ax4 = plt.subplot(2, 4, 4)
    ax4.plot(epochs, vit_history['val_acc'], 'b--o', label='ViT Val', linewidth=2)
    ax4.plot(epochs, clip_history['val_acc'], 'r--s', label='CLIP Val', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Validation Accuracy Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. ViT Confusion Matrix
    ax5 = plt.subplot(2, 4, 5)
    sns.heatmap(vit_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax5, cbar_kws={'label': 'Count'})
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    ax5.set_title('ViT Confusion Matrix')
    
    # 6. CLIP Confusion Matrix
    ax6 = plt.subplot(2, 4, 6)
    sns.heatmap(clip_cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, 
                yticklabels=class_names, ax=ax6, cbar_kws={'label': 'Count'})
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Actual')
    ax6.set_title('CLIP Confusion Matrix')
    
    # 7. Training Time Comparison
    ax7 = plt.subplot(2, 4, 7)
    avg_vit_time = np.mean(vit_history['epoch_time'])
    avg_clip_time = np.mean(clip_history['epoch_time'])
    ax7.bar(['ViT', 'CLIP'], [avg_vit_time, avg_clip_time], color=['blue', 'red'])
    ax7.set_ylabel('Time (seconds)')
    ax7.set_title('Average Training Time per Epoch')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Final Accuracy Comparison
    ax8 = plt.subplot(2, 4, 8)
    final_vit_acc = vit_history['val_acc'][-1]
    final_clip_acc = clip_history['val_acc'][-1]
    bars = ax8.bar(['ViT', 'CLIP'], [final_vit_acc, final_clip_acc], color=['blue', 'red'])
    ax8.set_ylabel('Accuracy (%)')
    ax8.set_title('Final Validation Accuracy')
    ax8.set_ylim([0, 100])
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plots saved to: {output_dir / 'model_comparison.png'}")


def main():
    # Dynamic path resolution - works on any machine
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    DATA_FOLDER_ENHANCED_RGB = PROJECT_ROOT / "data" / "Adani_MTF_Enhanced_224x224_RGB"
    OUTPUT_DIR = PROJECT_ROOT / "Model_Results"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 16  # Reduced for faster training on CPU
    NUM_EPOCHS = 10  # Reduced for faster completion
    LEARNING_RATE = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Use Enhanced RGB dataset
    DATA_FOLDER = DATA_FOLDER_ENHANCED_RGB
    dataset_type = "Enhanced_RGB"
    
    print(f"\n{'='*60}")
    print(f"Training on {dataset_type} Dataset")
    print(f"{'='*60}\n")
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = StockImageDataset(DATA_FOLDER, transform=val_transform)
    
    # Split dataset (80% train, 20% validation/test)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update transforms for training
    train_dataset.dataset.transform = train_transform
    
    print(f"\nDataset split:")
    print(f"  Training: {train_size} images")
    print(f"  Validation: {val_size} images")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize results dictionary
    results = {
        'dataset_type': dataset_type,
        'total_images': len(full_dataset),
        'train_images': train_size,
        'val_images': val_size,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }
    
    # =========================
    # Train Vision Transformer
    # =========================
    print("\n" + "="*60)
    print("VISION TRANSFORMER (ViT)")
    print("="*60)
    
    vit_model, vit_processor = load_vit_model(num_classes=3)
    
    if vit_model is not None:
        vit_model = vit_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(vit_model.parameters(), lr=LEARNING_RATE)
        
        vit_history, vit_best_acc = train_model(
            vit_model, train_loader, val_loader, criterion, optimizer, 
            device, NUM_EPOCHS, model_name="ViT"
        )
        
        # Evaluate ViT
        vit_acc, vit_report, vit_cm, vit_preds, vit_labels = evaluate_model(
            vit_model, val_loader, device, full_dataset.class_names, model_name="ViT"
        )
        
        results['vit'] = {
            'final_accuracy': vit_acc * 100,
            'best_val_accuracy': vit_best_acc,
            'classification_report': vit_report,
            'confusion_matrix': vit_cm.tolist()
        }
    else:
        print("Skipping ViT training due to loading error")
        vit_history, vit_cm = None, None
    
    # =========================
    # Train CLIP
    # =========================
    print("\n" + "="*60)
    print("CLIP MODEL")
    print("="*60)
    
    clip_model, clip_preprocess = load_clip_model(num_classes=3)
    
    if clip_model is not None:
        clip_model = clip_model.to(device)
        criterion = nn.CrossEntropyLoss()
        # Only train the classifier head, freeze CLIP encoder
        optimizer = optim.AdamW(clip_model.classifier.parameters(), lr=LEARNING_RATE)
        
        clip_history, clip_best_acc = train_model(
            clip_model, train_loader, val_loader, criterion, optimizer,
            device, NUM_EPOCHS, model_name="CLIP"
        )
        
        # Evaluate CLIP
        clip_acc, clip_report, clip_cm, clip_preds, clip_labels = evaluate_model(
            clip_model, val_loader, device, full_dataset.class_names, model_name="CLIP"
        )
        
        results['clip'] = {
            'final_accuracy': clip_acc * 100,
            'best_val_accuracy': clip_best_acc,
            'classification_report': clip_report,
            'confusion_matrix': clip_cm.tolist()
        }
    else:
        print("Skipping CLIP training due to loading error")
        clip_history, clip_cm = None, None
    
    # =========================
    # Generate Comparison Plots
    # =========================
    if vit_history and clip_history:
        plot_comparison_results(
            vit_history, clip_history, vit_cm, clip_cm,
            full_dataset.class_names, OUTPUT_DIR
        )
    
    # Save results to JSON
    results_file = OUTPUT_DIR / f'results_{dataset_type.lower()}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_file}")
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    print(f"Dataset: {dataset_type}")
    print(f"\nVision Transformer (ViT):")
    print(f"  Final Accuracy: {results['vit']['final_accuracy']:.2f}%")
    print(f"  Best Val Accuracy: {results['vit']['best_val_accuracy']:.2f}%")
    print(f"\nCLIP Model:")
    print(f"  Final Accuracy: {results['clip']['final_accuracy']:.2f}%")
    print(f"  Best Val Accuracy: {results['clip']['best_val_accuracy']:.2f}%")
    print("="*60)
    
    if results['vit']['final_accuracy'] > results['clip']['final_accuracy']:
        print(f"\n🏆 Winner: ViT by {results['vit']['final_accuracy'] - results['clip']['final_accuracy']:.2f}%")
    else:
        print(f"\n🏆 Winner: CLIP by {results['clip']['final_accuracy'] - results['vit']['final_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
