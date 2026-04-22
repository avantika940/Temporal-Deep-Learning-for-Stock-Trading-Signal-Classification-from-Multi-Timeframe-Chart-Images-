"""
Fixed Configuration for BiLSTM implementation - No Data Leakage

This configuration addresses the data leakage issues:
- Shorter sequences to reduce homogeneity
- More conservative training settings
- Proper evaluation metrics
"""

import os
from pathlib import Path


class FixedConfig:
    """Fixed Configuration class for BiLSTM models without data leakage"""
    
    # ========== Paths ==========
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "Adani_MTF_Images_224x224"
    RESULTS_DIR = Path(__file__).parent / "results_fixed"
    MODELS_DIR = RESULTS_DIR / "models"
    LOGS_DIR = RESULTS_DIR / "logs"
    VIZ_DIR = RESULTS_DIR / "visualizations"
    
    # Create directories
    for dir_path in [RESULTS_DIR, MODELS_DIR, LOGS_DIR, VIZ_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ========== Dataset Settings - FIXED ==========
    SEQUENCE_LENGTH = 10  # As requested by mentor - using 10 images for better temporal context
    IMAGE_SIZE = 224
    NUM_CLASSES = 3
    CLASS_NAMES = ['BUY', 'HOLD', 'SELL']
    
    # ========== Model Architecture ==========
    MODEL_TYPE = 'hybrid'
    
    # ViT settings (frozen feature extractor)
    VIT_MODEL = 'google/vit-base-patch16-224-in21k'
    VIT_DIM = 768
    FREEZE_VIT = True
    
    # BiLSTM settings - MORE CONSERVATIVE
    HIDDEN_DIM = 128  # Reduced from 256 to prevent overfitting
    NUM_LSTM_LAYERS = 2  # Reduced from 3
    BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.5  # Increased dropout
    
    # Attention settings
    ATTENTION_HEADS = 4  # Reduced from 8
    ATTENTION_DROPOUT = 0.3
    
    # Residual settings
    USE_LAYER_NORM = True
    RESIDUAL_DROPOUT = 0.3
    
    # Classifier settings
    CLASSIFIER_HIDDEN_DIM = 64  # Reduced from 256
    CLASSIFIER_DROPOUT = 0.5  # Increased dropout
    
    # ========== Training Settings - MORE CONSERVATIVE ==========
    BATCH_SIZE = 8  # Reduced from 16 for more stable training
    NUM_EPOCHS = 50  # Increased to allow for slower convergence
    LEARNING_RATE = 5e-5  # Reduced learning rate
    WEIGHT_DECAY = 1e-4  # Increased regularization
    
    # Optimizer
    OPTIMIZER = 'adamw'
    BETAS = (0.9, 0.999)
    
    # Learning rate scheduler - MORE AGGRESSIVE
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'reduce_on_plateau'
    SCHEDULER_FACTOR = 0.3  # More aggressive reduction
    SCHEDULER_PATIENCE = 2  # Shorter patience
    SCHEDULER_MIN_LR = 1e-7
    
    # ========== Regularization - INCREASED ==========
    GRADIENT_CLIP = 0.5  # More aggressive clipping
    USE_MIXUP = True  # Enable mixup for regularization
    MIXUP_ALPHA = 0.2
    LABEL_SMOOTHING = 0.1
    
    # ========== Early Stopping - MORE CONSERVATIVE ==========
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 8  # Longer patience for lower learning rate
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # ========== Data Loading ==========
    NUM_WORKERS = 0
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False
    
    # ========== Evaluation ==========
    EVAL_BATCH_SIZE = 16
    SAVE_BEST_ONLY = True
    SAVE_CONFUSION_MATRIX = True
    SAVE_TRAINING_CURVES = True
    SAVE_CLASSIFICATION_REPORT = True
    
    # ========== Realistic Expectations ==========
    # Expected accuracy for financial time series: 55-65%
    # Anything above 70% should be investigated for leakage
    EXPECTED_ACCURACY_RANGE = (55, 65)
    SUSPICIOUS_ACCURACY_THRESHOLD = 70
    
    # ========== Logging ==========
    LOG_INTERVAL = 10
    VERBOSE = True
    
    # ========== Reproducibility ==========
    RANDOM_SEED = 42
    DETERMINISTIC = True
    
    def __init__(self, **kwargs):
        """Initialize with optional overrides"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
    
    def to_dict(self):
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, Path):
                    result[key] = str(value)
                else:
                    result[key] = value
        return result
    
    def validate_results(self, accuracy):
        """Validate if results are realistic for financial data"""
        min_acc, max_acc = self.EXPECTED_ACCURACY_RANGE
        
        print(f"\nResult Validation:")
        print(f"Achieved accuracy: {accuracy:.2f}%")
        print(f"Expected range: {min_acc}-{max_acc}%")
        
        if accuracy > self.SUSPICIOUS_ACCURACY_THRESHOLD:
            print(f"⚠️  WARNING: Accuracy {accuracy:.2f}% is suspiciously high!")
            print(f"   Financial time series typically achieve {min_acc}-{max_acc}%")
            print(f"   Please check for data leakage issues.")
            return False
        elif accuracy < min_acc:
            print(f"⚠️  Low accuracy {accuracy:.2f}% - model may need tuning")
            return True
        else:
            print(f"✓ Realistic accuracy within expected range")
            return True


def get_fixed_config(**kwargs):
    """Get fixed configuration instance"""
    return FixedConfig(**kwargs)


if __name__ == "__main__":
    # Test the configuration
    config = FixedConfig()
    
    print("Fixed Configuration Test")
    print("=" * 50)
    print(f"Sequence Length: {config.SEQUENCE_LENGTH}")
    print(f"Hidden Dim: {config.HIDDEN_DIM}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Dropout: {config.LSTM_DROPOUT}")
    print(f"Expected Accuracy: {config.EXPECTED_ACCURACY_RANGE[0]}-{config.EXPECTED_ACCURACY_RANGE[1]}%")
    
    # Test validation
    print("\nTesting result validation:")
    config.validate_results(95.0)  # Should warn
    config.validate_results(62.0)  # Should pass
    config.validate_results(45.0)  # Should warn (low)
