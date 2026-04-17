from PIL import Image
import numpy as np
from pathlib import Path

# Dynamic path resolution - works on any machine
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
enhanced_dir = PROJECT_ROOT / 'data' / 'Adani_MTF_Enhanced_224x224'

print("Checking Enhanced 224x224 images:\n")

for category in ['BUY', 'HOLD', 'SELL']:
    img_path = list((enhanced_dir / category).glob('*.png'))[0]
    img = Image.open(img_path)
    arr = np.array(img)
    
    print(f"\n{category} - {img_path.name}:")
    print(f"  Mode: {img.mode}")
    print(f"  Size: {img.size}")
    print(f"  Array shape: {arr.shape}")
    print(f"  Value range: {arr.min()} to {arr.max()}")
    
    if len(arr.shape) == 3:
        # Check if all RGB channels are identical (grayscale stored as RGB)
        r_eq_g = np.all(arr[:,:,0] == arr[:,:,1])
        g_eq_b = np.all(arr[:,:,1] == arr[:,:,2])
        print(f"  All channels identical (grayscale): {r_eq_g and g_eq_b}")
    elif len(arr.shape) == 2:
        print(f"  Single channel (true grayscale/L mode)")
    
    print(f"  Unique values: {len(np.unique(arr))}")
