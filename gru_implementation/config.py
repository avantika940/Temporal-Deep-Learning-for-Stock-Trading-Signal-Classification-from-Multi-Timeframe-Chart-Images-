"""
Fixed Configuration file for GRU implementation - No Data Leakage

Contains conservative hyperparameters and settings to prevent overfitting
and ensure realistic results for financial time series prediction.
"""

import os
from pathlib import Path


class FixedConfig:
    """Fixed configuration class for GRU models with realistic expectations"""

    # ========== Paths ==========
    BASE_DIR = Path(__file__).parent.parent  # d:\MTECH
    DATA_DIR = BASE_DIR / "data" / "Adani_MTF_Images_224x224"
    RESULTS_DIR = Path(__file__).parent / "results_fixed"
    MODELS_DIR = RESULTS_DIR / "models"
    LOGS_DIR = RESULTS_DIR / "logs"
    VIZ_DIR = RESULTS_DIR / "visualizations"

    # Create directories
    for dir_path in [RESULTS_DIR, MODELS_DIR, LOGS_DIR, VIZ_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ========== Dataset Settings ==========
    SEQUENCE_LENGTH = 10         # As requested by mentor - using 10 images for better temporal context
    IMAGE_SIZE = 224
    NUM_CLASSES = 3
    CLASS_NAMES = ['BUY', 'HOLD', 'SELL']

    # ========== Model Architecture ==========
    MODEL_TYPE = 'deep'          # Start with simplest model

    # ViT settings (frozen feature extractor)
    VIT_MODEL = 'google/vit-base-patch16-224-in21k'
    VIT_DIM = 768
    FREEZE_VIT = True

    # GRU settings (conservative sizing)
    HIDDEN_DIM = 128             # Reduced from 256
    NUM_GRU_LAYERS = 2           # Reduced from 3
    BIDIRECTIONAL = True
    GRU_DROPOUT = 0.5            # Increased from 0.3

    # Attention settings
    ATTENTION_HEADS = 4          # Reduced from 8
    ATTENTION_DROPOUT = 0.3

    # Residual settings
    USE_LAYER_NORM = True
    RESIDUAL_DROPOUT = 0.3

    # Classifier settings (more regularization)
    CLASSIFIER_HIDDEN_DIM = 128  # Reduced from 256
    CLASSIFIER_DROPOUT = 0.5     # Increased from 0.3

    # ========== Training Settings ==========
    BATCH_SIZE = 16              # Increased for stability
    NUM_EPOCHS = 15              # Reduced from 20
    LEARNING_RATE = 5e-5         # Reduced from 1e-4
    WEIGHT_DECAY = 1e-4          # Increased regularization

    OPTIMIZER = 'adamw'
    BETAS = (0.9, 0.999)

    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'reduce_on_plateau'
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    SCHEDULER_MIN_LR = 1e-7

    # ========== Regularization ==========
    GRADIENT_CLIP = 0.5          # Reduced from 1.0
    LABEL_SMOOTHING = 0.1

    # ========== Early Stopping ==========
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.001

    # ========== Data Loading ==========
    NUM_WORKERS = 0
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False
    EVAL_BATCH_SIZE = 32

    # ========== Logging & Saving ==========
    LOG_INTERVAL = 10
    SAVE_BEST_ONLY = True
    VERBOSE = True

    # ========== Reproducibility ==========
    RANDOM_SEED = 42
    DETERMINISTIC = True

    # ========== Realistic Expectations ==========
    EXPECTED_ACCURACY_RANGE = (55, 65)      # Realistic for financial data
    SUSPICIOUS_ACCURACY_THRESHOLD = 70      # Flag if exceeded
    RANDOM_BASELINE = 33.33                 # 3-class random

    def __init__(self, **kwargs):
        """Initialize config and override with any provided kwargs"""
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

    def validate_results(self, accuracy: float) -> bool:
        """
        Validate if results are realistic for financial time series
        
        Args:
            accuracy: Validation/test accuracy to check
            
        Returns:
            True if realistic, False if suspicious
        """
        print(f"\nResult Validation:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Expected range: {self.EXPECTED_ACCURACY_RANGE[0]}-{self.EXPECTED_ACCURACY_RANGE[1]}%")
        print(f"  Random baseline: {self.RANDOM_BASELINE:.1f}%")
        
        if accuracy > self.SUSPICIOUS_ACCURACY_THRESHOLD:
            print(f"  ⚠️  SUSPICIOUS: Accuracy {accuracy:.2f}% > {self.SUSPICIOUS_ACCURACY_THRESHOLD}%")
            print(f"     This may indicate remaining data leakage!")
            return False
        elif accuracy < self.RANDOM_BASELINE + 5:
            print(f"  ⚠️  TOO LOW: Accuracy {accuracy:.2f}% barely above random")
            print(f"     Model may not be learning properly")
            return False
        elif self.EXPECTED_ACCURACY_RANGE[0] <= accuracy <= self.EXPECTED_ACCURACY_RANGE[1]:
            print(f"  ✅ REALISTIC: Accuracy in expected range")
            return True
        else:
            print(f"  ⚠️  MARGINAL: Accuracy outside typical range but not impossible")
            return True

    def validate_paths(self):
        """Validate that required paths exist"""
        if not self.DATA_DIR.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.DATA_DIR}\n"
                f"Please ensure the Adani MTF dataset exists at this location."
            )
        
        required_folders = ['BUY', 'HOLD', 'SELL']
        for folder in required_folders:
            folder_path = self.DATA_DIR / folder
            if not folder_path.exists():
                raise FileNotFoundError(
                    f"Required class folder not found: {folder_path}\n"
                    f"Dataset should contain BUY, HOLD, and SELL folders."
                )

    def print_config(self):
        """Print configuration summary"""
        print("=" * 60)
        print("FIXED GRU CONFIGURATION")
        print("=" * 60)
        print(f"Model Type: {self.MODEL_TYPE}")
        print(f"Sequence Length: {self.SEQUENCE_LENGTH}")
        print(f"Hidden Dim: {self.HIDDEN_DIM}")
        print(f"GRU Layers: {self.NUM_GRU_LAYERS}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Expected Accuracy: {self.EXPECTED_ACCURACY_RANGE[0]}-{self.EXPECTED_ACCURACY_RANGE[1]}%")
        print("=" * 60)


def get_config(**kwargs):
    """Get configuration instance with optional overrides"""
    return FixedConfig(**kwargs)


if __name__ == "__main__":
    config = FixedConfig()
    config.print_config()
    config.validate_paths()
