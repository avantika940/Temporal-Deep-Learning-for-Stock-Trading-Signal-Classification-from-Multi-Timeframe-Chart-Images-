"""
Image Processing Script
Enhanced upscaling of images to 224x224
"""

import os
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def regenerate_images_at_224(source_folder, target_folder):
    """
    Option 1: Regenerate images at native 224x224 resolution
    This recreates the charts/images at higher resolution from data
    
    Note: This requires knowing the source data or having vector graphics
    For demonstration, we'll enhance the existing images using advanced techniques
    """
    source_path = Path(source_folder)
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)
    
    subfolders = ['BUY', 'HOLD', 'SELL']
    
    # Count total images
    total_images = 0
    for subfolder in subfolders:
        subfolder_path = source_path / subfolder
        if subfolder_path.exists():
            total_images += len(list(subfolder_path.glob('*.png')))
    
    print(f"\n{'='*60}")
    print(f"Option 1: Smart Upscaling (36x36 → 224x224)")
    print(f"{'='*60}")
    print(f"Found {total_images} images to regenerate")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Method: LANCZOS + Sharpening\n")
    
    processed_count = 0
    
    with tqdm(total=total_images, desc="Regenerating images", unit="img") as pbar:
        for subfolder in subfolders:
            source_subfolder = source_path / subfolder
            target_subfolder = target_path / subfolder
            target_subfolder.mkdir(parents=True, exist_ok=True)
            
            if not source_subfolder.exists():
                continue
            
            for img_path in source_subfolder.glob('*.png'):
                try:
                    with Image.open(img_path) as img:
                        # Multi-step upscaling with sharpening
                        # Step 1: Resize to 224x224 with LANCZOS
                        upscaled = img.resize((224, 224), Image.LANCZOS)
                        
                        # Step 2: Apply sharpening to reduce blur
                        from PIL import ImageEnhance, ImageFilter
                        
                        # Sharpen the image
                        enhancer = ImageEnhance.Sharpness(upscaled)
                        sharpened = enhancer.enhance(2.0)  # Increase sharpness
                        
                        # Apply unsharp mask for better edge definition
                        final_img = sharpened.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
                        
                        # Increase contrast slightly
                        contrast = ImageEnhance.Contrast(final_img)
                        final_img = contrast.enhance(1.1)
                        
                        # Save to target folder
                        target_file = target_subfolder / img_path.name
                        final_img.save(target_file, 'PNG', optimize=True)
                        
                        processed_count += 1
                        pbar.update(1)
                        
                except Exception as e:
                    print(f"\nError processing {img_path.name}: {str(e)}")
                    pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"Regeneration Complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Output folder: {target_path}")
    print(f"{'='*60}\n")
    
    return processed_count


def create_visual_comparison(source_folder, padded_folder, upscaled_folder, output_file):
    """
    Create a visual comparison of the three approaches
    """
    source_path = Path(source_folder)
    padded_path = Path(padded_folder)
    upscaled_path = Path(upscaled_folder)
    
    # Get sample images from each category
    samples = []
    for subfolder in ['BUY', 'HOLD', 'SELL']:
        source_subfolder = source_path / subfolder
        if source_subfolder.exists():
            imgs = list(source_subfolder.glob('*.png'))
            if imgs:
                samples.append((subfolder, imgs[0].name))
                if len(samples) >= 3:
                    break
    
    if not samples:
        print("No sample images found for comparison")
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4*len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (subfolder, img_name) in enumerate(samples):
        # Original (upscaled for display)
        orig_path = source_path / subfolder / img_name
        if orig_path.exists():
            img = Image.open(orig_path)
            img_display = img.resize((224, 224), Image.NEAREST)  # Nearest neighbor to show pixelation
            axes[idx, 0].imshow(img_display)
            axes[idx, 0].set_title(f'Original 36x36\n(scaled for display)\n{subfolder}')
            axes[idx, 0].axis('off')
        
        # Padded
        padded_img_path = padded_path / subfolder / img_name
        if padded_img_path.exists():
            img = Image.open(padded_img_path)
            axes[idx, 1].imshow(img)
            axes[idx, 1].set_title(f'Padded to 224x224\n(Original preserved)\n{subfolder}')
            axes[idx, 1].axis('off')
        
        # Upscaled with enhancement
        upscaled_img_path = upscaled_path / subfolder / img_name
        if upscaled_img_path.exists():
            img = Image.open(upscaled_img_path)
            axes[idx, 2].imshow(img)
            axes[idx, 2].set_title(f'Smart Upscaled 224x224\n(Enhanced)\n{subfolder}')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisual comparison saved to: {output_file}")


def main():
    # Dynamic path resolution - works on any machine
    base_folder = Path(__file__).parent.parent.parent.resolve()
    source_folder = base_folder / "data" / "Adani MTF Image Under Compress"
    
    # Output folder
    upscaled_folder = base_folder / "data" / "Adani_MTF_Enhanced_224x224"
    
    # Check if source exists
    if not source_folder.exists():
        print(f"Error: Source folder not found: {source_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"IMAGE PROCESSING - ENHANCED UPSCALING")
    print(f"{'='*60}")
    print(f"Processing {source_folder}")
    print(f"{'='*60}\n")
    
    # Smart upscaling with enhancement
    regenerate_images_at_224(source_folder, upscaled_folder)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Enhanced Images: {upscaled_folder}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
