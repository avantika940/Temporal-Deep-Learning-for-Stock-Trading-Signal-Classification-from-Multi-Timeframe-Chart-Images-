"""
Image Resizing Script for Vision Transformer (ViT) / CLIP Models
Resizes images from 36x36 to 224x224 pixels using high-quality interpolation
"""

import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def resize_images(source_folder, target_folder, target_size=(224, 224)):
    """
    Resize all PNG images from source folder to target folder.
    
    Args:
        source_folder: Path to the source folder containing images
        target_folder: Path to the target folder for resized images
        target_size: Tuple of (width, height) for the target size
    """
    source_path = Path(source_folder)
    target_path = Path(target_folder)
    
    # Create target folder if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Define subfolders to process
    subfolders = ['BUY', 'HOLD', 'SELL']
    
    # Count total images first
    total_images = 0
    for subfolder in subfolders:
        subfolder_path = source_path / subfolder
        if subfolder_path.exists():
            total_images += len(list(subfolder_path.glob('*.png')))
    
    print(f"Found {total_images} images to resize")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Target size: {target_size[0]}x{target_size[1]} pixels\n")
    
    # Process each subfolder
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    with tqdm(total=total_images, desc="Resizing images", unit="img") as pbar:
        for subfolder in subfolders:
            source_subfolder = source_path / subfolder
            target_subfolder = target_path / subfolder
            
            # Create target subfolder
            target_subfolder.mkdir(parents=True, exist_ok=True)
            
            if not source_subfolder.exists():
                print(f"Warning: Subfolder {subfolder} not found in source")
                continue
            
            # Process all PNG images in the subfolder
            for img_path in source_subfolder.glob('*.png'):
                try:
                    # Open the image
                    with Image.open(img_path) as img:
                        # Get original size
                        original_size = img.size
                        
                        # Resize using LANCZOS (high-quality resampling filter)
                        # LANCZOS is better than BICUBIC for upscaling
                        resized_img = img.resize(target_size, Image.LANCZOS)
                        
                        # Save to target folder with same filename
                        target_file = target_subfolder / img_path.name
                        resized_img.save(target_file, 'PNG', optimize=True)
                        
                        processed_count += 1
                        pbar.update(1)
                        
                except Exception as e:
                    error_count += 1
                    print(f"\nError processing {img_path.name}: {str(e)}")
                    pbar.update(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Resizing Complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_count} images")
    if error_count > 0:
        print(f"Errors encountered: {error_count} images")
    print(f"Output folder: {target_path}")
    print(f"{'='*60}")


def main():
    # Dynamic path resolution - works on any machine
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    
    # Define source and target folders
    source_folder = PROJECT_ROOT / "data" / "Adani MTF Image Under Compress"
    target_folder = PROJECT_ROOT / "data" / "Adani_MTF_Images_224x224"
    
    # Target size for ViT/CLIP models
    target_size = (224, 224)
    
    # Check if source folder exists
    if not source_folder.exists():
        print(f"Error: Source folder not found: {source_folder}")
        return
    
    # Resize images
    resize_images(source_folder, target_folder, target_size)


if __name__ == "__main__":
    main()
