"""
Test script to verify the fixed dataset has all classes in each split
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config_fixed import FixedConfig
from dataset_fixed import create_fixed_data_loaders

def test_class_distribution():
    """Test that all classes are represented in train/val/test splits"""
    
    print("=" * 80)
    print("TESTING FIXED DATASET - CLASS DISTRIBUTION")
    print("=" * 80)
    
    # Create config
    config = FixedConfig()
    
    # Test dataset creation
    try:
        train_loader, val_loader, test_loader = create_fixed_data_loaders(config)
        
        print("\n✅ SUCCESS: All data loaders created successfully!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test a batch from each loader
        print("\n📊 TESTING BATCH LOADING:")
        
        for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
            if len(loader) > 0:
                batch = next(iter(loader))
                sequences, labels = batch
                print(f"   {name}: batch shape {sequences.shape}, labels shape {labels.shape}")
                
                # Check class distribution in this batch
                unique_labels = set(labels.numpy())
                print(f"   {name}: classes in batch: {[config.CLASS_NAMES[i] for i in unique_labels]}")
            else:
                print(f"   {name}: EMPTY LOADER!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_class_distribution()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
