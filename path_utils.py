"""
Path utilities for the MTECH project.

Provides dynamic path resolution that works across different machines
without hardcoded paths.
"""

from pathlib import Path


def get_project_root():
    """
    Get the project root directory dynamically.
    
    Returns:
        Path: The root directory of the MTECH project
    """
    # This file is in the project root, so return its parent directory
    return Path(__file__).parent.resolve()


# Project root directory - works on any machine
PROJECT_ROOT = get_project_root()

# Common data directories
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_MULTI_PROCESSED_DIR = PROJECT_ROOT / "data_multi_processed"

# Common output directories
MODEL_RESULTS_DIR = PROJECT_ROOT / "Model_Results"
MULTI_STOCK_RESULTS_DIR = PROJECT_ROOT / "multi_stock_results"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Specific data folders
ADANI_MTF_IMAGES_224x224 = DATA_DIR / "Adani_MTF_Images_224x224"
ADANI_MTF_ENHANCED_224x224 = DATA_DIR / "Adani_MTF_Enhanced_224x224"
ADANI_MTF_ENHANCED_RGB = DATA_DIR / "Adani_MTF_Enhanced_224x224_RGB"
ADANI_MTF_IMAGE_COMPRESS = DATA_DIR / "Adani MTF Image Under Compress"


def get_script_dir():
    """Get the directory where the calling script is located."""
    import inspect
    frame = inspect.currentframe().f_back
    caller_file = frame.f_globals.get('__file__')
    if caller_file:
        return Path(caller_file).parent.resolve()
    return Path.cwd()


def ensure_dir(path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path object or string path to directory
        
    Returns:
        Path: The directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    print("="*80)
    print("MTECH Project Path Configuration")
    print("="*80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\nData Directories:")
    print(f"  DATA_DIR:                {DATA_DIR}")
    print(f"  DATASETS_DIR:            {DATASETS_DIR}")
    print(f"  DATA_MULTI_PROCESSED:    {DATA_MULTI_PROCESSED_DIR}")
    print(f"\nOutput Directories:")
    print(f"  MODEL_RESULTS:           {MODEL_RESULTS_DIR}")
    print(f"  MULTI_STOCK_RESULTS:     {MULTI_STOCK_RESULTS_DIR}")
    print(f"\nSpecific Data Folders:")
    print(f"  Adani 224x224:           {ADANI_MTF_IMAGES_224x224}")
    print(f"  Adani Enhanced:          {ADANI_MTF_ENHANCED_224x224}")
    print(f"  Adani Enhanced RGB:      {ADANI_MTF_ENHANCED_RGB}")
    print("="*80)
