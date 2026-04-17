"""
Configuration file for BiGRU implementation

Contains all hyperparameters and settings for training and evaluation.
Mirrors the BiLSTM config but tuned for GRU (fewer gates → can afford
slightly larger hidden dims or more epochs for the same wall-clock budget).
"""

import os
from pathlib import Path


class Config:
    """Configuration class for BiGRU models"""

    # ========== Paths ==========
    # Dynamically determine project root (parent of gru_implementation folder)
    BASE_DIR = Path(__file__).parent.parent  # d:\MTECH (or wherever the project is)
    DATA_DIR = BASE_DIR / "data" / "Adani_MTF_Images_224x224"
    RESULTS_DIR = Path(__file__).parent / "results"
    MODELS_DIR = RESULTS_DIR / "models"
    LOGS_DIR = RESULTS_DIR / "logs"
    VIZ_DIR = RESULTS_DIR / "visualizations"

    # Create directories
    for dir_path in [RESULTS_DIR, MODELS_DIR, LOGS_DIR, VIZ_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ========== Dataset Settings ==========
    SEQUENCE_LENGTH = 10          # Cross-class timeline window size
    IMAGE_SIZE = 224
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    NUM_CLASSES = 3
    CLASS_NAMES = ['BUY', 'HOLD', 'SELL']

    # ========== Model Architecture ==========
    MODEL_TYPE = 'hybrid'         # deep | attention | residual | hybrid | pyramidal

    # ViT settings (frozen feature extractor — identical to BiLSTM baseline)
    VIT_MODEL = 'google/vit-base-patch16-224-in21k'
    VIT_DIM = 768
    FREEZE_VIT = True

    # BiGRU settings  (GRU has no cell state → ~33 % fewer params than same-size LSTM)
    HIDDEN_DIM = 256
    NUM_GRU_LAYERS = 3
    BIDIRECTIONAL = True
    GRU_DROPOUT = 0.3

    # Attention settings (for attention-based models)
    ATTENTION_HEADS = 8
    ATTENTION_DROPOUT = 0.2

    # Residual settings
    USE_LAYER_NORM = True
    RESIDUAL_DROPOUT = 0.2

    # Classifier settings
    CLASSIFIER_HIDDEN_DIM = 256
    CLASSIFIER_DROPOUT = 0.3

    # ========== Training Settings ==========
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    OPTIMIZER = 'adamw'           # adam | adamw | sgd
    BETAS = (0.9, 0.999)
    MOMENTUM = 0.9                # For SGD

    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'reduce_on_plateau'   # reduce_on_plateau | cosine | step
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    SCHEDULER_MIN_LR = 1e-7

    T_MAX = 10
    ETA_MIN = 1e-6
    STEP_SIZE = 5
    GAMMA = 0.5

    # ========== Regularization ==========
    GRADIENT_CLIP = 1.0
    USE_MIXUP = False
    MIXUP_ALPHA = 0.2
    LABEL_SMOOTHING = 0.1

    # ========== Early Stopping ==========
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.001

    # ========== Data Loading ==========
    NUM_WORKERS = 0               # 0 for Windows compatibility
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False

    # ========== Logging ==========
    LOG_INTERVAL = 10
    SAVE_BEST_ONLY = True
    VERBOSE = True

    # ========== Evaluation ==========
    EVAL_BATCH_SIZE = 16
    SAVE_CONFUSION_MATRIX = True
    SAVE_TRAINING_CURVES = True
    SAVE_CLASSIFICATION_REPORT = True

    # ========== Reproducibility ==========
    RANDOM_SEED = 42
    DETERMINISTIC = True

    # ========== Device ==========
    DEVICE = 'cuda'

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")

    def to_dict(self):
        """Convert config to dictionary (JSON-serialisable)"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, Path):
                    result[key] = str(value)
                else:
                    result[key] = value
        return result

    def validate_paths(self):
        """Validate that required paths exist and provide helpful error messages"""
        if not self.DATA_DIR.exists():
            raise FileNotFoundError(
                f"\n{'='*80}\n"
                f"ERROR: Data directory not found!\n"
                f"Expected: {self.DATA_DIR}\n"
                f"{'='*80}\n"
                f"Please ensure your data is placed in the correct location.\n"
                f"Current BASE_DIR: {self.BASE_DIR}\n"
                f"If your project is in a different location, the paths will\n"
                f"automatically adjust based on where this script is run from.\n"
                f"{'='*80}"
            )
        
        # Check if data directory has the expected structure
        expected_classes = ['BUY', 'HOLD', 'SELL']
        missing_classes = [cls for cls in expected_classes if not (self.DATA_DIR / cls).exists()]
        if missing_classes:
            raise FileNotFoundError(
                f"\n{'='*80}\n"
                f"ERROR: Missing class folders in data directory!\n"
                f"Missing: {missing_classes}\n"
                f"Data directory: {self.DATA_DIR}\n"
                f"Expected structure:\n"
                f"  {self.DATA_DIR}/\n"
                f"    ├── BUY/\n"
                f"    ├── HOLD/\n"
                f"    └── SELL/\n"
                f"{'='*80}"
            )
        
        return True

    def print_config(self):
        """Print all configuration parameters"""
        print("\n" + "=" * 80)
        print("BiGRU CONFIGURATION")
        print("=" * 80)
        print(f"\n📁 PATHS:")
        print(f"  Data Directory: {self.DATA_DIR}")
        print(f"  Results Directory: {self.RESULTS_DIR}")
        print(f"\n📊 DATASET:")
        print(f"  Sequence Length: {self.SEQUENCE_LENGTH}")
        print(f"  Image Size: {self.IMAGE_SIZE}")
        print(f"  Classes: {self.NUM_CLASSES} {self.CLASS_NAMES}")
        print(f"  Train/Val Split: {self.TRAIN_SPLIT}/{self.VAL_SPLIT}")
        print(f"\n🧠 MODEL:")
        print(f"  Model Type: {self.MODEL_TYPE}")
        print(f"  ViT Model: {self.VIT_MODEL} (frozen={self.FREEZE_VIT})")
        print(f"  Hidden Dim: {self.HIDDEN_DIM}")
        print(f"  GRU Layers: {self.NUM_GRU_LAYERS}")
        print(f"  Bidirectional: {self.BIDIRECTIONAL}")
        print(f"\n⚙️  TRAINING:")
        print(f"  Epochs: {self.NUM_EPOCHS}")
        print(f"  Batch Size: {self.BATCH_SIZE}")
        print(f"  Learning Rate: {self.LEARNING_RATE}")
        print(f"  Optimizer: {self.OPTIMIZER}")
        print(f"  Early Stopping: {self.EARLY_STOPPING} (patience={self.EARLY_STOPPING_PATIENCE})")
        print("=" * 80)


def get_config(model_type: str) -> Config:
    """Return a config pre-set for the requested model type."""
    configs = {
        'deep':      Config(MODEL_TYPE='deep'),
        'attention': Config(MODEL_TYPE='attention'),
        'residual':  Config(MODEL_TYPE='residual'),
        'hybrid':    Config(MODEL_TYPE='hybrid'),
        'pyramidal': Config(MODEL_TYPE='pyramidal'),
    }
    if model_type not in configs:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Choose from: {list(configs.keys())}")
    return configs[model_type]
