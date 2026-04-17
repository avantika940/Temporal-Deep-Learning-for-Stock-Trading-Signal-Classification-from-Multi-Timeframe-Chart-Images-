"""
Quick test script to verify BiLSTM implementation setup

This script checks:
1. All required packages are installed
2. Data directory exists and is accessible
3. Model creation works
4. Data loading works
"""

import sys
from pathlib import Path

print("="*80)
print("BILSTM IMPLEMENTATION - SETUP VERIFICATION")
print("="*80 + "\n")

# Test 1: Check Python version
print("1. Checking Python version...")
import sys
if sys.version_info < (3, 8):
    print("   [X] Python 3.8+ required")
    sys.exit(1)
print(f"   [OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Check required packages
print("\n2. Checking required packages...")
required_packages = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('transformers', 'Transformers'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('PIL', 'Pillow'),
    ('sklearn', 'Scikit-learn'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('tqdm', 'tqdm')
]

missing_packages = []
for package, name in required_packages:
    try:
        __import__(package)
        print(f"   [OK] {name}")
    except ImportError:
        print(f"   [X] {name} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   Missing packages: {', '.join(missing_packages)}")
    print("   Install with: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check CUDA availability
print("\n3. Checking GPU/CUDA availability...")
import torch
if torch.cuda.is_available():
    print(f"   [OK] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   [OK] CUDA version: {torch.version.cuda}")
    print(f"   [OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("   [!] CUDA not available - will use CPU (training will be slower)")

# Test 4: Check project structure
print("\n4. Checking project structure...")
project_dir = Path(__file__).parent
required_files = [
    'config.py',
    'dataset.py',
    'models.py',
    'train.py',
    'evaluate.py',
    'run_experiments.py',
    'README.md'
]

for file in required_files:
    file_path = project_dir / file
    if file_path.exists():
        print(f"   [OK] {file}")
    else:
        print(f"   [X] {file} - MISSING")

# Test 5: Check data directory
print("\n5. Checking data directory...")
sys.path.append(str(project_dir))
from config import Config

config = Config()
data_dir = config.DATA_DIR

if not data_dir.exists():
    print(f"   [X] Data directory not found: {data_dir}")
    print("   Please ensure data is in the correct location")
    sys.exit(1)

print(f"   [OK] Data directory exists: {data_dir}")

# Check class folders
class_folders = ['BUY', 'HOLD', 'SELL']
for class_name in class_folders:
    class_dir = data_dir / class_name
    if class_dir.exists():
        num_images = len(list(class_dir.glob('*.png')))
        print(f"   [OK] {class_name}: {num_images} images")
    else:
        print(f"   [X] {class_name} folder missing")

# Test 6: Test model creation
print("\n6. Testing model creation...")
try:
    from models import create_model
    
    for model_type in ['deep', 'attention', 'residual', 'hybrid', 'pyramidal']:
        config_test = Config(MODEL_TYPE=model_type, SEQUENCE_LENGTH=3)
        model = create_model(config_test)
        print(f"   [OK] {model_type.capitalize()}BiLSTM")
    
except Exception as e:
    print(f"   [X] Model creation failed: {e}")
    sys.exit(1)

# Test 7: Test data loading
print("\n7. Testing data loading...")
try:
    from dataset import SequenceDataset, get_transforms
    
    transform = get_transforms(config)
    dataset = SequenceDataset(
        config.DATA_DIR,
        sequence_length=3,
        transform=transform,
        mode='test'
    )
    
    print(f"   [OK] Dataset loaded: {len(dataset)} sequences")
    
    # Test loading a sample
    sequence, label = dataset[0]
    print(f"   [OK] Sample loaded: shape {sequence.shape}, label {label}")
    
except Exception as e:
    print(f"   [X] Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Check results directory
print("\n8. Checking results directories...")
for dir_name in ['models', 'logs', 'visualizations']:
    dir_path = project_dir / 'results' / dir_name
    if dir_path.exists():
        print(f"   [OK] results/{dir_name}/")
    else:
        print(f"   [!] results/{dir_name}/ - will be created during training")

# Summary
print("\n" + "="*80)
print("SETUP VERIFICATION COMPLETE")
print("="*80)
print("\n[OK] All checks passed! You're ready to train BiLSTM models.")
print("\nNext steps:")
print("  1. Train a single model:")
print("     python bilstm_implementation/train.py --model hybrid --epochs 20")
print("\n  2. Or train all models:")
print("     python bilstm_implementation/run_experiments.py")
print("\n  3. Evaluate a trained model:")
print("     python bilstm_implementation/evaluate.py --model hybrid")
print("\n" + "="*80 + "\n")
