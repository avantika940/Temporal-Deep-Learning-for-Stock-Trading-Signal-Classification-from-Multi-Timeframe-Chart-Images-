"""
Quick Training Script for ViT and CLIP on Enhanced RGB Images
Optimized for CPU with reduced complexity
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
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.images)} images")
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
    """Create simple transforms"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy * 100, all_preds, all_labels


def main():
    print("\n" + "="*60)
    print("QUICK TRAINING: ViT and CLIP on Enhanced RGB Images")
    print("="*60)
    
    # Dynamic path resolution - works on any machine
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    DATA_FOLDER = PROJECT_ROOT / "data" / "Adani_MTF_Enhanced_224x224_RGB"
    OUTPUT_DIR = PROJECT_ROOT / "Model_Results"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 8  # Small batch for CPU
    NUM_EPOCHS = 5  # Quick training
    LEARNING_RATE = 1e-4
    
    device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    transform = create_transforms()
    full_dataset = StockImageDataset(DATA_FOLDER, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images\n")
    
    results = {}
    
    # =========================
    # Train ViT
    # =========================
    print("\n" + "="*60)
    print("TRAINING VISION TRANSFORMER (ViT)")
    print("="*60)
    
    try:
        from transformers import ViTForImageClassification
        
        print("Loading ViT model...")
        vit_model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        vit_model = vit_model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(vit_model.parameters(), lr=LEARNING_RATE)
        
        print(f"\nTraining for {NUM_EPOCHS} epochs...")
        best_vit_acc = 0
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            train_loss, train_acc = train_epoch(vit_model, train_loader, criterion, optimizer, device)
            val_acc, val_preds, val_labels = evaluate(vit_model, val_loader, device)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_vit_acc:
                best_vit_acc = val_acc
        
        # Final evaluation
        print("\nFinal ViT Evaluation:")
        final_vit_acc, vit_preds, vit_labels = evaluate(vit_model, val_loader, device)
        vit_cm = confusion_matrix(vit_labels, vit_preds)
        vit_report = classification_report(vit_labels, vit_preds, target_names=full_dataset.class_names)
        
        print(f"Final Accuracy: {final_vit_acc:.2f}%")
        print(f"Best Val Accuracy: {best_vit_acc:.2f}%")
        print("\nClassification Report:")
        print(vit_report)
        print("\nConfusion Matrix:")
        print(vit_cm)
        
        results['vit'] = {
            'final_accuracy': final_vit_acc,
            'best_val_accuracy': best_vit_acc,
            'classification_report': vit_report,
            'confusion_matrix': vit_cm.tolist()
        }
        
    except Exception as e:
        print(f"Error training ViT: {e}")
        results['vit'] = {'error': str(e)}
    
    # =========================
    # Train CLIP
    # =========================
    print("\n" + "="*60)
    print("TRAINING CLIP MODEL")
    print("="*60)
    
    try:
        import clip
        
        print("Loading CLIP model...")
        clip_model_base, clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Add classification head
        class CLIPClassifier(nn.Module):
            def __init__(self, clip_model, num_classes):
                super().__init__()
                self.clip_model = clip_model
                self.classifier = nn.Linear(512, num_classes)
                
            def forward(self, images):
                with torch.no_grad():
                    features = self.clip_model.encode_image(images)
                features = features.float()
                return self.classifier(features)
        
        clip_classifier = CLIPClassifier(clip_model_base, 3).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(clip_classifier.classifier.parameters(), lr=LEARNING_RATE)
        
        print(f"\nTraining for {NUM_EPOCHS} epochs...")
        best_clip_acc = 0
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            train_loss, train_acc = train_epoch(clip_classifier, train_loader, criterion, optimizer, device)
            val_acc, val_preds, val_labels = evaluate(clip_classifier, val_loader, device)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_clip_acc:
                best_clip_acc = val_acc
        
        # Final evaluation
        print("\nFinal CLIP Evaluation:")
        final_clip_acc, clip_preds, clip_labels = evaluate(clip_classifier, val_loader, device)
        clip_cm = confusion_matrix(clip_labels, clip_preds)
        clip_report = classification_report(clip_labels, clip_preds, target_names=full_dataset.class_names)
        
        print(f"Final Accuracy: {final_clip_acc:.2f}%")
        print(f"Best Val Accuracy: {best_clip_acc:.2f}%")
        print("\nClassification Report:")
        print(clip_report)
        print("\nConfusion Matrix:")
        print(clip_cm)
        
        results['clip'] = {
            'final_accuracy': final_clip_acc,
            'best_val_accuracy': best_clip_acc,
            'classification_report': clip_report,
            'confusion_matrix': clip_cm.tolist()
        }
        
    except Exception as e:
        print(f"Error training CLIP: {e}")
        results['clip'] = {'error': str(e)}
    
    # Save results
    results_file = OUTPUT_DIR / 'results_enhanced_rgb.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    if 'error' not in results.get('vit', {}) and 'error' not in results.get('clip', {}):
        print(f"\nVision Transformer (ViT):")
        print(f"  Final Accuracy: {results['vit']['final_accuracy']:.2f}%")
        print(f"  Best Val Accuracy: {results['vit']['best_val_accuracy']:.2f}%")
        
        print(f"\nCLIP Model:")
        print(f"  Final Accuracy: {results['clip']['final_accuracy']:.2f}%")
        print(f"  Best Val Accuracy: {results['clip']['best_val_accuracy']:.2f}%")
        
        if results['vit']['final_accuracy'] > results['clip']['final_accuracy']:
            print(f"\n🏆 Winner: ViT by {results['vit']['final_accuracy'] - results['clip']['final_accuracy']:.2f}%")
        else:
            print(f"\n🏆 Winner: CLIP by {results['clip']['final_accuracy'] - results['vit']['final_accuracy']:.2f}%")
    
    print(f"\nResults saved to: {results_file}")
    print("="*60)


if __name__ == "__main__":
    main()
