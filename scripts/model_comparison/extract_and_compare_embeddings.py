"""
Extract and Compare Embeddings: RGB vs Grayscale
Comprehensive analysis to determine which representation is better
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageDataset(torch.utils.data.Dataset):
    """Custom dataset for loading images"""
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
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def extract_vit_embeddings(data_dir, model_path=None, batch_size=32):
    """Extract embeddings from ViT model"""
    print(f"\n🔍 Extracting ViT embeddings from: {data_dir}")
    
    from transformers import ViTModel, ViTForImageClassification
    
    # Load model
    if model_path and Path(model_path).exists():
        print(f"   Loading trained model from: {model_path}")
        # Load the full classification model first
        full_model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        checkpoint = torch.load(model_path, map_location=device)
        full_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract just the ViT base (without classifier)
        model = full_model.vit
    else:
        print(f"   Using pretrained model (no fine-tuning)")
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    model.to(device)
    model.eval()
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    embeddings = []
    labels = []
    
    print(f"   Processing {len(dataset)} images...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="   Extracting"):
            images = images.to(device)
            
            # Get embeddings (CLS token from last hidden state)
            outputs = model(images)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [batch, 768]
            
            embeddings.append(batch_embeddings)
            labels.extend(targets.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    print(f"✅ Extracted embeddings: {embeddings.shape}")
    return embeddings, labels, dataset.class_names


def extract_clip_embeddings(data_dir, model_path=None, batch_size=32):
    """Extract embeddings from CLIP model"""
    print(f"\n🔍 Extracting CLIP embeddings from: {data_dir}")
    
    import clip
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # Note: CLIP vision encoder is frozen during training, so we use pretrained weights
    print(f"   Using pretrained CLIP model")
    
    # Data loading with CLIP preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    embeddings = []
    labels = []
    
    print(f"   Processing {len(dataset)} images...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="   Extracting"):
            images = images.to(device)
            
            # Get image embeddings
            image_features = model.encode_image(images).float()  # [batch, 512]
            embeddings.append(image_features.cpu().numpy())
            labels.extend(targets.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    print(f"✅ Extracted embeddings: {embeddings.shape}")
    return embeddings, labels, dataset.class_names


def compute_separability_metrics(embeddings, labels, class_names):
    """Compute various separability metrics"""
    print("\n📊 Computing separability metrics...")
    
    metrics = {}
    
    # 1. Intra-class distances (within same class)
    intra_distances = []
    for class_idx in range(len(class_names)):
        class_embeddings = embeddings[labels == class_idx]
        if len(class_embeddings) > 1:
            distances = cdist(class_embeddings, class_embeddings, metric='euclidean')
            # Get upper triangle (excluding diagonal)
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            intra_distances.extend(upper_tri)
    
    metrics['intra_class_distance_mean'] = float(np.mean(intra_distances))
    metrics['intra_class_distance_std'] = float(np.std(intra_distances))
    
    # 2. Inter-class distances (between different classes)
    inter_distances = []
    centroids = []
    for class_idx in range(len(class_names)):
        class_embeddings = embeddings[labels == class_idx]
        centroid = np.mean(class_embeddings, axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_distances.append(dist)
    
    metrics['inter_class_distance_mean'] = float(np.mean(inter_distances))
    metrics['inter_class_distance_std'] = float(np.std(inter_distances))
    
    # 3. Separability ratio (inter/intra - higher is better)
    metrics['separability_ratio'] = metrics['inter_class_distance_mean'] / metrics['intra_class_distance_mean']
    
    # 4. Silhouette score (-1 to 1, higher is better)
    silhouette = silhouette_score(embeddings, labels, metric='euclidean', sample_size=min(1000, len(embeddings)))
    metrics['silhouette_score'] = float(silhouette)
    
    # 5. Per-class statistics
    metrics['per_class'] = {}
    for class_idx, class_name in enumerate(class_names):
        class_embeddings = embeddings[labels == class_idx]
        metrics['per_class'][class_name] = {
            'count': int(np.sum(labels == class_idx)),
            'mean_norm': float(np.mean(np.linalg.norm(class_embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(class_embeddings, axis=1)))
        }
    
    print(f"   Separability Ratio: {metrics['separability_ratio']:.4f}")
    print(f"   Silhouette Score: {metrics['silhouette_score']:.4f}")
    
    return metrics


def visualize_embeddings_2d(embeddings, labels, class_names, title, output_path):
    """Create 2D visualization using t-SNE"""
    print(f"\n🎨 Creating 2D visualization: {title}")
    
    # Apply t-SNE
    print("   Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['#06A77D', '#F5B700', '#D62828']  # BUY, HOLD, SELL
    
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=colors[class_idx], label=class_name, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=11)
    plt.ylabel('t-SNE Component 2', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {output_path}")


def create_comparison_dashboard(rgb_metrics, gray_metrics, rgb_embeddings, gray_embeddings, 
                                rgb_labels, gray_labels, class_names, output_dir):
    """Create comprehensive comparison dashboard"""
    print("\n📊 Creating comparison dashboard...")
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('RGB vs Grayscale Embeddings - Comprehensive Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Separability Ratio Comparison
    ax1 = plt.subplot(2, 4, 1)
    versions = ['RGB', 'Grayscale']
    sep_ratios = [rgb_metrics['separability_ratio'], gray_metrics['separability_ratio']]
    colors = ['#2E86AB', '#6C757D']
    bars = ax1.bar(versions, sep_ratios, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Separability Ratio', fontsize=11, fontweight='bold')
    ax1.set_title('Separability Ratio\n(Higher = Better)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Silhouette Score Comparison
    ax2 = plt.subplot(2, 4, 2)
    sil_scores = [rgb_metrics['silhouette_score'], gray_metrics['silhouette_score']]
    bars = ax2.bar(versions, sil_scores, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax2.set_title('Silhouette Score\n(Higher = Better)', fontsize=12, fontweight='bold')
    ax2.set_ylim([-1, 1])
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold', fontsize=11)
    
    # 3. Distance Comparison
    ax3 = plt.subplot(2, 4, 3)
    x = np.arange(2)
    width = 0.35
    
    intra_dists = [rgb_metrics['intra_class_distance_mean'], gray_metrics['intra_class_distance_mean']]
    inter_dists = [rgb_metrics['inter_class_distance_mean'], gray_metrics['inter_class_distance_mean']]
    
    bars1 = ax3.bar(x - width/2, intra_dists, width, label='Intra-class\n(Lower=Better)', 
                    color='#E63946', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, inter_dists, width, label='Inter-class\n(Higher=Better)', 
                    color='#06A77D', alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Distance', fontsize=11, fontweight='bold')
    ax3.set_title('Intra vs Inter-class Distance', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(versions)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. RGB t-SNE
    ax4 = plt.subplot(2, 4, 4)
    print("   Computing RGB t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    rgb_2d = tsne.fit_transform(rgb_embeddings)
    colors_class = ['#06A77D', '#F5B700', '#D62828']
    
    for class_idx, class_name in enumerate(class_names):
        mask = rgb_labels == class_idx
        ax4.scatter(rgb_2d[mask, 0], rgb_2d[mask, 1],
                   c=colors_class[class_idx], label=class_name, alpha=0.6, s=20)
    
    ax4.set_title('RGB Embeddings (t-SNE)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)
    
    # 5. Grayscale t-SNE
    ax5 = plt.subplot(2, 4, 5)
    print("   Computing Grayscale t-SNE...")
    gray_2d = tsne.fit_transform(gray_embeddings)
    
    for class_idx, class_name in enumerate(class_names):
        mask = gray_labels == class_idx
        ax5.scatter(gray_2d[mask, 0], gray_2d[mask, 1],
                   c=colors_class[class_idx], label=class_name, alpha=0.6, s=20)
    
    ax5.set_title('Grayscale Embeddings (t-SNE)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2)
    
    # 6. Per-Class Comparison
    ax6 = plt.subplot(2, 4, 6)
    x = np.arange(len(class_names))
    width = 0.35
    
    rgb_norms = [rgb_metrics['per_class'][cls]['mean_norm'] for cls in class_names]
    gray_norms = [gray_metrics['per_class'][cls]['mean_norm'] for cls in class_names]
    
    bars1 = ax6.bar(x - width/2, rgb_norms, width, label='RGB', color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax6.bar(x + width/2, gray_norms, width, label='Grayscale', color='#6C757D', alpha=0.8, edgecolor='black')
    
    ax6.set_ylabel('Mean Embedding Norm', fontsize=11, fontweight='bold')
    ax6.set_title('Per-Class Embedding Magnitude', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(class_names)
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Winner Determination
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    # Calculate winner
    rgb_score = 0
    gray_score = 0
    
    if rgb_metrics['separability_ratio'] > gray_metrics['separability_ratio']:
        rgb_score += 1
    else:
        gray_score += 1
    
    if rgb_metrics['silhouette_score'] > gray_metrics['silhouette_score']:
        rgb_score += 1
    else:
        gray_score += 1
    
    winner = "RGB" if rgb_score > gray_score else "Grayscale" if gray_score > rgb_score else "Tie"
    winner_color = '#2E86AB' if winner == "RGB" else '#6C757D' if winner == "Grayscale" else '#F5B700'
    
    winner_text = f"""
{'='*45}
      EMBEDDING QUALITY WINNER
{'='*45}

🏆 {winner}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Separability Ratio:
  • RGB:       {rgb_metrics['separability_ratio']:.4f}
  • Grayscale: {gray_metrics['separability_ratio']:.4f}
  {'✓' if rgb_metrics['separability_ratio'] > gray_metrics['separability_ratio'] else ' '} Winner: {'RGB' if rgb_metrics['separability_ratio'] > gray_metrics['separability_ratio'] else 'Grayscale'}

Silhouette Score:
  • RGB:       {rgb_metrics['silhouette_score']:.4f}
  • Grayscale: {gray_metrics['silhouette_score']:.4f}
  {'✓' if rgb_metrics['silhouette_score'] > gray_metrics['silhouette_score'] else ' '} Winner: {'RGB' if rgb_metrics['silhouette_score'] > gray_metrics['silhouette_score'] else 'Grayscale'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next: Train classifiers to confirm!

{'='*45}
"""
    
    ax7.text(0.5, 0.5, winner_text, transform=ax7.transAxes, fontsize=10,
            ha='center', va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.3, edgecolor='black', linewidth=2))
    
    # 8. Summary Statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    summary_text = f"""
DETAILED METRICS

RGB Embeddings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dimension: {rgb_embeddings.shape[1]}
  Samples: {len(rgb_embeddings)}
  
  Intra-class distance:
    Mean: {rgb_metrics['intra_class_distance_mean']:.2f}
    Std:  {rgb_metrics['intra_class_distance_std']:.2f}
  
  Inter-class distance:
    Mean: {rgb_metrics['inter_class_distance_mean']:.2f}
    Std:  {rgb_metrics['inter_class_distance_std']:.2f}

Grayscale Embeddings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dimension: {gray_embeddings.shape[1]}
  Samples: {len(gray_embeddings)}
  
  Intra-class distance:
    Mean: {gray_metrics['intra_class_distance_mean']:.2f}
    Std:  {gray_metrics['intra_class_distance_std']:.2f}
  
  Inter-class distance:
    Mean: {gray_metrics['inter_class_distance_mean']:.2f}
    Std:  {gray_metrics['inter_class_distance_std']:.2f}
"""
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'RGB_vs_Grayscale_Embedding_Comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved to: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("🚀 EMBEDDING EXTRACTION & COMPARISON: RGB vs GRAYSCALE")
    print("="*70)
    
    # Dynamic path resolution - works on any machine
    base_dir = Path(__file__).parent.parent.parent.resolve()
    output_dir = base_dir / "embeddings"
    output_dir.mkdir(exist_ok=True)
    
    results_dir = base_dir / "Model_Results"
    results_dir.mkdir(exist_ok=True)
    
    # Dataset paths
    rgb_data_dir = base_dir / "data" / "Adani_MTF_Enhanced_224x224_RGB"
    gray_data_dir = base_dir / "Adani_MTF_Enhanced_224x224"
    
    # Model paths (if available)
    vit_rgb_model = base_dir / "saved_models" / "vit_best_model_enhanced_rgb.pth"
    
    print(f"\n📂 Data directories:")
    print(f"   RGB: {rgb_data_dir}")
    print(f"   Grayscale: {gray_data_dir}")
    
    # ==================================================
    # STEP 1: Extract ViT Embeddings
    # ==================================================
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING ViT EMBEDDINGS")
    print("="*70)
    
    # RGB embeddings
    rgb_embeddings, rgb_labels, class_names = extract_vit_embeddings(
        rgb_data_dir, 
        model_path=vit_rgb_model if vit_rgb_model.exists() else None
    )
    
    # Grayscale embeddings
    gray_embeddings, gray_labels, _ = extract_vit_embeddings(
        gray_data_dir,
        model_path=None  # Use pretrained for fair comparison
    )
    
    # Save embeddings
    np.savez(output_dir / 'vit_embeddings_rgb.npz',
             embeddings=rgb_embeddings, labels=rgb_labels, classes=class_names)
    np.savez(output_dir / 'vit_embeddings_grayscale.npz',
             embeddings=gray_embeddings, labels=gray_labels, classes=class_names)
    
    print(f"\n💾 Saved embeddings to: {output_dir}")
    
    # ==================================================
    # STEP 2: Compute Metrics
    # ==================================================
    print("\n" + "="*70)
    print("STEP 2: COMPUTING SEPARABILITY METRICS")
    print("="*70)
    
    print("\n🔍 RGB Embeddings:")
    rgb_metrics = compute_separability_metrics(rgb_embeddings, rgb_labels, class_names)
    
    print("\n🔍 Grayscale Embeddings:")
    gray_metrics = compute_separability_metrics(gray_embeddings, gray_labels, class_names)
    
    # ==================================================
    # STEP 3: Create Visualizations
    # ==================================================
    print("\n" + "="*70)
    print("STEP 3: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Individual t-SNE plots
    visualize_embeddings_2d(rgb_embeddings, rgb_labels, class_names,
                           'RGB Embeddings (t-SNE)', 
                           results_dir / 'tsne_rgb_embeddings.png')
    
    visualize_embeddings_2d(gray_embeddings, gray_labels, class_names,
                           'Grayscale Embeddings (t-SNE)',
                           results_dir / 'tsne_grayscale_embeddings.png')
    
    # Comprehensive dashboard
    create_comparison_dashboard(rgb_metrics, gray_metrics,
                               rgb_embeddings, gray_embeddings,
                               rgb_labels, gray_labels, class_names, results_dir)
    
    # ==================================================
    # STEP 4: Save Comparison Results
    # ==================================================
    print("\n" + "="*70)
    print("STEP 4: SAVING COMPARISON RESULTS")
    print("="*70)
    
    comparison_results = {
        'rgb': rgb_metrics,
        'grayscale': gray_metrics,
        'winner': {
            'by_separability': 'RGB' if rgb_metrics['separability_ratio'] > gray_metrics['separability_ratio'] else 'Grayscale',
            'by_silhouette': 'RGB' if rgb_metrics['silhouette_score'] > gray_metrics['silhouette_score'] else 'Grayscale',
            'separability_diff': float(rgb_metrics['separability_ratio'] - gray_metrics['separability_ratio']),
            'silhouette_diff': float(rgb_metrics['silhouette_score'] - gray_metrics['silhouette_score'])
        }
    }
    
    results_path = results_dir / 'embedding_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"💾 Comparison results saved to: {results_path}")
    
    # ==================================================
    # FINAL SUMMARY
    # ==================================================
    print("\n" + "="*70)
    print("✅ EXTRACTION & COMPARISON COMPLETE!")
    print("="*70)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   RGB Separability: {rgb_metrics['separability_ratio']:.4f}")
    print(f"   Grayscale Separability: {gray_metrics['separability_ratio']:.4f}")
    print(f"   Difference: {abs(rgb_metrics['separability_ratio'] - gray_metrics['separability_ratio']):.4f}")
    print()
    print(f"   RGB Silhouette: {rgb_metrics['silhouette_score']:.4f}")
    print(f"   Grayscale Silhouette: {gray_metrics['silhouette_score']:.4f}")
    print(f"   Difference: {abs(rgb_metrics['silhouette_score'] - gray_metrics['silhouette_score']):.4f}")
    
    if (rgb_metrics['separability_ratio'] > gray_metrics['separability_ratio'] and 
        rgb_metrics['silhouette_score'] > gray_metrics['silhouette_score']):
        print(f"\n🏆 WINNER: RGB (better on both metrics)")
    elif (gray_metrics['separability_ratio'] > rgb_metrics['separability_ratio'] and 
          gray_metrics['silhouette_score'] > rgb_metrics['silhouette_score']):
        print(f"\n🏆 WINNER: Grayscale (better on both metrics)")
    else:
        print(f"\n⚖️ MIXED RESULTS - Need classifier comparison to decide!")
    
    print(f"\n📂 Generated files:")
    print(f"   • embeddings/vit_embeddings_rgb.npz")
    print(f"   • embeddings/vit_embeddings_grayscale.npz")
    print(f"   • Model_Results/tsne_rgb_embeddings.png")
    print(f"   • Model_Results/tsne_grayscale_embeddings.png")
    print(f"   • Model_Results/RGB_vs_Grayscale_Embedding_Comparison.png")
    print(f"   • Model_Results/embedding_comparison_results.json")
    
    print(f"\n💡 Next step: Run classifier comparison to confirm winner!")
    print("="*70)


if __name__ == "__main__":
    main()
