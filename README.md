# Multi-Stock MTF Classification System

A comprehensive deep learning framework for multi-stock Market-Technical-Fundamental (MTF) image classification using hybrid neural networks and ensemble methods.

## Table of Contents

- [Overview](#overview)
- [Visual Documentation](#visual-documentation)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Performance Analysis](#performance-analysis)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a novel approach to stock market trend prediction by combining visual pattern recognition with technical indicators. The system processes Market-Technical-Fundamental (MTF) images and generates multi-class predictions for stock movements (BUY/HOLD/SELL).

### Key Features

- **Hybrid Feature Engineering**: Combines frozen Vision Transformer (ViT) embeddings with derived technical indicators
- **Multi-Model Architecture**: Implements HybridBiGRU and HybridBiLSTM for sequence learning
- **Ensemble Methods**: Soft, weighted, and hard voting classifiers for improved performance
- **Chronological Splitting**: Ensures no temporal data leakage in train/validation/test splits
- **Multi-Stock Support**: Trained and evaluated on 4 major stocks (COAL India, HCL, Maruti, Apollo)

## Visual Documentation

This project includes comprehensive visual documentation through Mermaid diagrams that illustrate the research methodology, architectures, and results. All diagrams are available in the [`Figures.md`](Figures.md) file.

### Available Diagrams

The visual documentation covers all aspects of the research pipeline:

#### **Research Flow & Methodology**
- Complete research pipeline overview (Figure 1)
- Five-step training methodology (Figure 2)
- Data preprocessing and feature extraction workflow (Figures 3-4)
- Chronological data splitting strategy (Figure 5)

#### **Model Architectures**
- Vision Transformer (ViT) feature extraction pipeline (Figure 6)
- Technical indicator calculation flowchart (Figure 7)
- HybridBiGRU architecture diagram (Figure 8)
- HybridBiLSTM architecture diagram (Figure 9)
- Ensemble methods comparison (Figure 10)

#### **Results & Analysis**
- Multi-stock performance comparison (Figure 11)
- Training time analysis across stocks (Figure 12)
- Confusion matrix layouts for all models (Figures 13-20)
- Ensemble performance improvements (Figure 21)

### Using the Diagrams

The Mermaid diagrams can be rendered in several ways:

1. **GitHub/GitLab**: Automatically rendered when viewing `Figures.md` on GitHub
2. **VS Code**: Install the Mermaid Preview extension
3. **Online Editors**: Copy code to [Mermaid Live Editor](https://mermaid.live)
4. **Documentation Tools**: Integrate with Sphinx, MkDocs, or similar tools
5. **Research Papers**: Export as SVG/PNG for LaTeX documents

#### **Quick Access**
- 📊 [View All Figures](Figures.md)
- 🔄 [Research Pipeline Diagram](Figures.md#figure-1-complete-research-pipeline)
- 🏗️ [Model Architectures](Figures.md#figure-6-vision-transformer-vit-feature-extraction)
- 📈 [Performance Results](Figures.md#figure-11-multi-stock-performance-comparison)

> **Note**: The diagrams are designed to complement the research paper draft in [`Q1_Journal_Paper_Complete_Draft.md`](Q1_Journal_Paper_Complete_Draft.md), providing visual representations of all key concepts and results discussed in the academic paper.

## Architecture

### Feature Vector Composition

Each data point consists of a 773-dimensional feature vector:

- **Visual Features (768-dim)**: Frozen ViT-Base embeddings from google/vit-base-patch16-224-in21k
- **Technical Indicators (5-dim)**: Derived from class-label signal sequences
  1. RSI-proxy: Average gain ratio [0,1]
  2. MACD-proxy: Fast-Slow EMA difference [-1,1]
  3. MACD-signal: EMA of MACD-proxy [-1,1]
  4. Momentum: Label sequence momentum [-1,1]
  5. Volume-weighted Signal: Volume-adjusted signal [-1,1]

### Model Architecture

```
Input: Sequential MTF Images (seq_len=10)
  ↓
ViT Feature Extraction (Frozen) → 768-dim embeddings
  ↓
Technical Indicator Calculation → 5-dim indicators
  ↓
Feature Concatenation → 773-dim vectors
  ↓
Sequence Learning (BiGRU/BiLSTM) → Hidden representations
  ↓
Classification Head → BUY/HOLD/SELL predictions
```

## Dataset

### Data Sources

- **Stock Coverage**: 4 major Indian stocks (COAL India, HCL, Maruti, Apollo)
- **Image Format**: 224x224 RGB MTF images
- **Temporal Range**: Chronologically ordered market data
- **Class Distribution**: Balanced BUY/HOLD/SELL classifications

### Data Preprocessing

- Images resized and normalized for ViT input
- Chronological sorting by filename ID
- Sliding window approach (stride=1, window=10)
- No random shuffling to preserve temporal integrity

## Models

### 1. HybridBiGRU
- Bidirectional GRU with 128 hidden units
- Dropout: 0.3
- Dense output layer with softmax activation

### 2. HybridBiLSTM
- Bidirectional LSTM with 128 hidden units
- Dropout: 0.3
- Dense output layer with softmax activation

### 3. Ensemble Methods
- **Soft Voting**: Averages predicted probabilities
- **Weighted Voting**: Performance-weighted averaging
- **Hard Voting**: Majority vote classification

## Results

### Overall Performance Summary

| Metric | HybridBiGRU | HybridBiLSTM |
|--------|-------------|-------------|
| **Single-Stock Best** | 58.82% test | **68.42% test** |
| **Multi-Stock Average** | 55.91% test | **59.27% test** |
| **Validation Average** | 65.84% | 63.80% |
| **Training Efficiency** | 6.37 min | 5.47 min |

> **Note**: For detailed realistic performance analysis, see [`REALISTIC_RESULTS_SUMMARY.md`](REALISTIC_RESULTS_SUMMARY.md)

### Stock-Specific Performance

| Stock | Model | Test Acc | Precision | Recall | F1 Score | Training Time |
|-------|-------|----------|-----------|---------|----------|---------------|
| **COAL India** | BiGRU | 66.23% | 0.475 | 0.662 | 0.544 | 658.84 min |
| **COAL India** | BiLSTM | 35.65% | 0.207 | 0.357 | 0.242 | 22.70 min |
| **HCL** | BiGRU | 65.04% | 0.652 | 0.650 | 0.596 | 9.02 min |
| **HCL** | BiLSTM | 47.85% | 0.545 | 0.479 | 0.475 | 10.18 min |
| **Maruti** | BiGRU | 57.54% | 0.570 | 0.575 | 0.573 | 2.96 min |
| **Maruti** | BiLSTM | 64.80% | 0.632 | 0.648 | 0.580 | 3.02 min |
| **Apollo** | BiGRU | 57.70% | 0.579 | 0.577 | 0.544 | 2.86 min |
| **Apollo** | BiLSTM | 58.31% | 0.537 | 0.583 | 0.520 | 3.12 min |

### Ensemble Performance

| Stock | Ensemble Accuracy | Precision | Recall | F1 Score | Improvement |
|-------|------------------|-----------|---------|----------|-------------|
| **Apollo** | 65.56% | 0.663 | 0.656 | 0.611 | +7.86% |
| **Maruti** | 58.66% | 0.559 | 0.587 | 0.564 | -6.14% |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install Pillow
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-stock-mtf-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare data directories:
```bash
mkdir -p data_multi_processed/{COAL_India_MTF,HCL_MTF,MARUTI_MTF,Apollo_MTF}
mkdir -p embeddings
mkdir -p multi_stock_results
```

## Usage

### 1. Feature Extraction

Extract ViT embeddings for all stock images:

```bash
python finetune_vit_extract_embeddings.py
```

### 2. Training Pipeline

Run the complete multi-stock training pipeline:

```bash
python multi_stock_pipeline.py
```

### 3. Visualization Generation

Generate comprehensive results visualizations:

```bash
python generate_visualizations_fixed.py
```

### 4. Individual Analysis

Generate individual confusion matrices:

```bash
python generate_individual_matrices.py
```

## Project Structure

```
multi-stock-mtf-classification/
├── README.md                        # This documentation
├── requirements.txt                 # Python dependencies
├── Q1_Journal_Paper_Complete_Draft.md  # Research paper draft
├── Figures.md                       # Mermaid diagrams for all figures
├── multi_stock_pipeline.py          # Main training pipeline
├── finetune_vit_extract_embeddings.py  # Feature extraction
├── generate_visualizations_fixed.py    # Results visualization
├── generate_individual_matrices.py     # Individual analysis
├── path_utils.py                     # Utility functions
│
├── data_multi_processed/            # Preprocessed stock data
│   ├── COAL_India_MTF/
│   ├── HCL_MTF/
│   ├── MARUTI_MTF/
│   └── Apollo_MTF/
│
├── embeddings/                      # ViT embeddings cache
│   ├── COAL_India_MTF_frozen_rgb.npz
│   ├── HCL_MTF_frozen_rgb.npz
│   ├── MARUTI_MTF_frozen_rgb.npz
│   └── Apollo_MTF_frozen_rgb.npz
│
├── multi_stock_results/             # Results and visualizations
│   ├── results_filtered_4stocks.json
│   ├── summary.csv
│   ├── confusion_matrices_all_stocks.png
│   ├── performance_comparison_analysis.png
│   ├── ensemble_performance_analysis.png
│   └── complete_results_dashboard.png
│
├── bilstm_implementation/           # BiLSTM specific code
├── gru_implementation/             # BiGRU specific code
└── Model_Results/                  # Historical results
```

## Technical Details

### Data Split Strategy

- **Chronological Splitting**: 64% train, 16% validation, 20% test
- **No Temporal Leakage**: Strict chronological ordering preserved
- **Sequence Generation**: Sliding window with stride=1, length=10

### Training Configuration

- **Epochs**: 80 (fixed)
- **Batch Size**: 32
- **Learning Rate**: 3e-4 with CosineAnnealingLR
- **Optimizer**: Adam
- **Loss Function**: Cross-entropy with class weighting
- **Sequence Length**: 10 frames

### Feature Engineering

- **ViT Model**: google/vit-base-patch16-224-in21k (frozen)
- **Technical Indicators**: Derived from label sequences
- **Normalization**: Min-max scaling for technical indicators
- **Class Weighting**: Inverse frequency weighting

## Performance Analysis

### Key Insights

1. **Model Comparison**:
   - BiGRU shows higher validation accuracy but variable test performance
   - BiLSTM demonstrates more consistent test performance across stocks

2. **Stock-Specific Patterns**:
   - COAL India: Significant training time variation (BiGRU: 658 min vs BiLSTM: 23 min)
   - Apollo & Maruti: Fast training with moderate performance
   - HCL: Best overall performance with BiGRU

3. **Ensemble Benefits**:
   - Apollo shows significant improvement (+7.86%) with ensemble methods
   - Maruti shows slight degradation, suggesting model diversity issues

### Training Efficiency

- **Fastest Training**: Apollo and Maruti (~3 minutes per model)
- **Slowest Training**: COAL India BiGRU (658 minutes)
- **Average Training Time**: 82.6 minutes per model

## Future Work

### Potential Improvements

1. **Architecture Enhancements**:
   - Implement attention mechanisms
   - Explore transformer-based sequence models
   - Add multi-scale temporal features

2. **Feature Engineering**:
   - Incorporate additional technical indicators
   - Add fundamental analysis features
   - Implement dynamic feature selection

3. **Data Augmentation**:
   - Temporal data augmentation techniques
   - Synthetic minority oversampling
   - Cross-stock transfer learning

4. **Ensemble Optimization**:
   - Learn optimal ensemble weights
   - Implement stacking methods
   - Add model diversity metrics

### Scalability

- Extend to more stock symbols
- Implement distributed training
- Add real-time inference capabilities
- Cloud deployment optimization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Vision Transformer implementation from Hugging Face Transformers
- Technical indicator calculations inspired by TA-Lib
- Deep learning frameworks: PyTorch and scikit-learn

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the maintainers.

### Documentation Resources

- 📖 **Complete Research Paper**: [`Q1_Journal_Paper_Complete_Draft.md`](Q1_Journal_Paper_Complete_Draft.md) *(excluded from git)*
- 📊 **Visual Documentation**: [`Figures.md`](Figures.md) - All Mermaid diagrams with updated realistic results
- 📈 **Realistic Results Analysis**: [`REALISTIC_RESULTS_SUMMARY.md`](REALISTIC_RESULTS_SUMMARY.md) - Corrected performance metrics
- 🔧 **Technical Implementation**: This README and code documentation
- � **Results Analysis**: [`multi_stock_results/`](multi_stock_results/) directory

---

**Note**: This project is for research and educational purposes. Stock market predictions should not be used as the sole basis for investment decisions.
