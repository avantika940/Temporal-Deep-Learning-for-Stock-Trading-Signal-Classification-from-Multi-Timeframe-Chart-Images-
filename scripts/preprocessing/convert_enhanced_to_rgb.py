"""
Convert grayscale enhanced images to proper RGB format
Since the images are stock charts (grayscale data), we'll convert them to RGB
by applying a colormap for better feature extraction by the model
"""

from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm


def convert_grayscale_to_rgb(source_folder, output_folder, colormap='viridis'):
    """
    Convert grayscale images to RGB using a colormap
    This helps models better distinguish features in the charts
    
    Args:
        source_folder: Folder with grayscale images (stored as RGB but all channels identical)
        output_folder: Folder to save true RGB images
        colormap: Matplotlib colormap to use ('viridis', 'plasma', 'inferno', 'hot', etc.)
    """
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subfolders = ['BUY', 'HOLD', 'SELL']
    cmap = cm.get_cmap(colormap)
    
    # Count total images
    total_images = 0
    for subfolder in subfolders:
        subfolder_path = source_path / subfolder
        if subfolder_path.exists():
            total_images += len(list(subfolder_path.glob('*.png')))
    
    print(f"\n{'='*60}")
    print(f"Converting Grayscale Images to RGB")
    print(f"{'='*60}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Colormap: {colormap}")
    print(f"Total images: {total_images}\n")
    
    processed_count = 0
    
    with tqdm(total=total_images, desc="Converting images", unit="img") as pbar:
        for subfolder in subfolders:
            source_subfolder = source_path / subfolder
            output_subfolder = output_path / subfolder
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            if not source_subfolder.exists():
                continue
            
            for img_path in source_subfolder.glob('*.png'):
                try:
                    # Load image
                    img = Image.open(img_path)
                    
                    # Convert to grayscale properly (take just one channel since they're all identical)
                    img_gray = img.convert('L')
                    img_array = np.array(img_gray)
                    
                    # Normalize to 0-1 range
                    img_normalized = img_array / 255.0
                    
                    # Apply colormap to convert to RGB
                    img_colored = cmap(img_normalized)
                    
                    # Convert back to 0-255 range and remove alpha channel
                    img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)
                    
                    # Save as PIL Image
                    img_rgb_pil = Image.fromarray(img_rgb, mode='RGB')
                    output_file = output_subfolder / img_path.name
                    img_rgb_pil.save(output_file, 'PNG', optimize=True)
                    
                    processed_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError processing {img_path.name}: {str(e)}")
                    pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Output folder: {output_path}")
    print(f"{'='*60}\n")
    
    return processed_count


if __name__ == "__main__":
    # Dynamic path resolution - works on any machine
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    
    # Source folder with grayscale images
    source = PROJECT_ROOT / "data" / "Adani_MTF_Enhanced_224x224"
    
    # Output folder for RGB images
    output = PROJECT_ROOT / "data" / "Adani_MTF_Enhanced_224x224_RGB"
    
    # Convert using viridis colormap (good for data visualization)
    # Other options: 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm'
    print("Starting conversion with 'viridis' colormap...")
    convert_grayscale_to_rgb(str(source), str(output), colormap='viridis')
    
    print("\nNOTE: If you prefer to keep the images as grayscale but properly formatted,")
    print("the model will still work since .convert('RGB') duplicates the channel.")
    print("Using a colormap can help the model distinguish features better.")
