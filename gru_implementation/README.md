# BiGRU Implementation — Complete Documentation

## 1. How Image Order Is Decided

### The Filename IS the Timestamp

Every image in the dataset has a globally unique sequential ID baked into its filename:

```
BUY_0.png    → position 0 in market time
BUY_1.png    → position 1
...
SELL_23.png  → position 23
BUY_24.png   → position 24
SELL_25.png  → position 25
HOLD_28.png  → position 28
...
HOLD_1745.png → position 1745 (last)
```

The IDs run **0 → 1745** with no gaps — exactly **1,746 images**, each with a unique position.

- The **class prefix** (BUY_, HOLD_, SELL_) tells you what signal that moment in time represented
- The **number** tells you when it happened

All three class folders are merged into one list and sorted by this ID. The actual merged timeline looks like:

| ID Range | Class | Images |
|----------|-------|--------|
| 0 – 22   | BUY   | BUY_0.png … BUY_22.png |
| 23       | SELL  | SELL_23.png |
| 24       | BUY   | BUY_24.png |
| 25 – 27  | SELL  | SELL_25.png … SELL_27.png |
| 28 – 38  | HOLD  | HOLD_28.png … HOLD_38.png |
| 39       | SELL  | SELL_39.png |
| …        | …     | … |

This is the original **market chronology** — the signal changed as: BUY → SELL → BUY → SELL → HOLD → SELL → … as time progressed.

---

## 2. How BUY / HOLD / SELL Classification Works

Adani's stock data was converted to MTF (Multi-Timeframe) chart images and sorted into 3 folders:

```
BUY/   → 585 images
HOLD/  → 586 images
SELL/  → 575 images
```

**The folder the image is in = its class.** Simple as that. The class was pre-decided when the dataset was created.

### What Does the Model Do?

The model **never sees the folder name**. It only sees the image pixels. It must learn from the images:
> *"What pattern in this chart means BUY? What means SELL?"*

During training, for every image:
1. Model looks at the pixels → makes a guess (BUY / HOLD / SELL)
2. Code checks: is that guess correct compared to the folder it came from?
3. If wrong → model adjusts itself (backpropagation) to do better next time

### The Sequence Label

When 10 images are grouped into a sequence, the label of the **whole sequence** is the folder of the **10th (last) image**:

```
Photo 1:  BUY folder  ┐
Photo 2:  BUY folder  │
Photo 3:  SELL folder │  ← these 9 are just "context"
...                   │
Photo 9:  BUY folder  ┘
Photo 10: HOLD folder ← this one decides the sequence label = HOLD
```

**The model's job:** *"Look at 10 consecutive MTF charts → predict what signal the latest chart is giving."*

---

## 3. How BiGRU Works

### What is a GRU?

A **Gated Recurrent Unit (GRU)** is a type of neural network designed to process sequences.  
It uses **2 gates** to control how information flows through time:

| Gate | What it does |
|------|-------------|
| **Reset gate** | How much of the past to forget |
| **Update gate** | How much of the old state to keep vs new information |

GRU has **~33% fewer parameters** than LSTM (no separate cell state / output gate) → faster training, comparable accuracy.

### What is "Bi" (Bidirectional)?

One GRU reads the 10-frame sequence **left → right** (past to present).  
A second GRU reads it **right → left** (present to past).  
Their outputs are **concatenated** at each time step → the model sees both past AND future context.

If `HIDDEN_DIM = 256`, each direction gives 256 → combined output is **512-dim** per time step.

---

## 4. The Full Pipeline (Step by Step)

```
10 MTF chart images  (each 224×224 pixels, RGB)
        ↓
  Frozen ViT (google/vit-base-patch16-224-in21k)
        ↓
  CLS token embedding per image  →  10 × 768 features
        ↓
  BiGRU layer(s) process the sequence
        ↓
  Pooling / Attention over time steps
        ↓
  Fully Connected classifier
        ↓
  3 output scores  →  argmax  →  BUY / HOLD / SELL
```

### Why Frozen ViT?

ViT (Vision Transformer) is used only as a **feature extractor**. It converts each 224×224 image into a 768-dimensional vector (CLS token) that captures rich visual patterns. By keeping it frozen, we:
- Don't need to retrain ViT (saves GPU memory and time)
- Reuse powerful pre-trained ImageNet knowledge
- Focus training only on the temporal (BiGRU) part

---

## 5. Five Model Architectures

### Model 1: DeepBiGRU (Baseline)

```
ViT → BiGRU(256) → BiGRU(256) → BiGRU(128) → last time-step → FC → 3 classes
```

- 3 stacked BiGRU layers with decreasing sizes
- Uses only the **last time step** for prediction
- Dropout between layers for regularisation
- **Val Accuracy: 58.91%**

---

### Model 2: AttentionBiGRU

```
ViT → 2-layer BiGRU → Self-Attention (weighted sum over all 10 steps) → FC → 3 classes
```

- Adds **additive self-attention pooling** over all 10 time steps
- Instead of just the last frame, a learned weighted sum of all steps is used
- The model learns *which time steps matter most*
- **Val Accuracy: 60.06%**

---

### Model 3: ResidualBiGRU

```
ViT → Linear Projection → [BiGRU + Skip Connection + LayerNorm] × 3 → last step → FC
```

- Adds **residual (skip) connections**: `output = GRU(x) + x`
- Layer normalisation after each block
- Prevents vanishing gradients in deep layers
- **Val Accuracy: 54.60%**

---

### Model 4: HybridBiGRU ⭐ (Best)

```
ViT → Linear Projection
    → [BiGRU + Skip + LayerNorm] × 3     (local temporal patterns)
    → Multi-Head Self-Attention (8 heads) (global context across all 10 steps)
    → Gated Fusion (learned blend of GRU + Attention)
    → Mean Pool over sequence
    → FC → 3 classes
```

**Gated Fusion** — the key innovation:
```
gate = Sigmoid(Linear([gru_out, attn_out]))
fused = gate × gru_out + (1 - gate) × attn_out
```
The gate is a learned value between 0 and 1 that controls how much of GRU vs Attention to use — per position, per sample.

- **Val Accuracy: 72.70%** ← best result

---

### Model 5: PyramidalBiGRU

```
ViT → BiGRU(512) → BiGRU(256) → BiGRU(128)
    → Temporal Pyramid Pooling [scales: 1, 2, 5]
    → Concatenate all scales → FC → 3 classes
```

- GRU hidden dims shrink like a pyramid (coarse to fine)
- **Temporal pyramid pooling** captures patterns at multiple time scales
- **Val Accuracy: 49.71%**

---

## 6. Training Results Summary

| Model | Train Acc | Val Acc | Epochs | Time (min) |
|-------|-----------|---------|--------|------------|
| DeepBiGRU | 63.93% | 58.91% | 5 | 18.74 |
| AttentionBiGRU | 62.35% | 60.06% | 5 | 19.11 |
| ResidualBiGRU | 57.24% | 54.60% | 5 | 17.38 |
| **HybridBiGRU** ⭐ | **72.71%** | **72.70%** | **5** | **17.39** |
| PyramidalBiGRU | — | 49.71% | 1 | — |

> Baseline reference (Step-4 Temporal Transformer): 61.21% val accuracy  
> **HybridBiGRU improves over baseline by +11.49 percentage points**

### Epoch-by-Epoch Progress of HybridBiGRU

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 48.31% | 53.16% |
| 2 | 54.21% | 56.03% |
| 3 | 60.55% | 56.03% |
| 4 | 68.61% | 70.40% |
| 5 | 72.71% | 72.70% |

---

## 7. How Accuracy Is Calculated

### During Training (every epoch)

```python
out = model(seqs)            # raw logits: [batch, 3]
_, pred = torch.max(out, 1)  # pick class with highest score
correct += (pred == labels).sum().item()
accuracy = 100 * correct / total
```

### During Final Evaluation

```python
acc = accuracy_score(targets, preds) * 100   # sklearn
```

Also computed:
- **Precision** — of all samples predicted as class X, how many actually are X
- **Recall** — of all actual class X samples, how many did the model find
- **F1-score** — harmonic mean of precision & recall
- All use `average='weighted'` to handle class imbalance

### Labels Are Pre-Loaded — Not Looked Up After Prediction

The true labels come from filenames at dataset load time and are stored in memory alongside the images. When the DataLoader iterates, it gives `(images, true_label)` together in each batch. The model sees only images; the true label is compared after prediction — no second lookup into the dataset.

---

## 8. Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 10 frames |
| Image Size | 224 × 224 |
| ViT Model | google/vit-base-patch16-224-in21k (frozen) |
| ViT Embedding Dim | 768 |
| Hidden Dim (BiGRU) | 256 |
| GRU Dropout | 0.3 |
| Attention Heads | 8 |
| Classifier Hidden Dim | 256 |
| Batch Size | 8 |
| Epochs | 5–10 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| LR Scheduler | ReduceOnPlateau |
| Gradient Clipping | 1.0 |
| Train / Val Split | 80% / 20% |

---

## 9. Project File Structure

```
gru_implementation/
├── config.py          — All hyperparameters and paths
├── dataset.py         — Timeline merging, sliding window, DataLoader
├── models.py          — 5 BiGRU architectures
├── train.py           — Training loop, early stopping, checkpointing
├── evaluate.py        — Metrics, confusion matrix, training curves
├── run_seq10_all_models.py — Run all 5 models sequentially
└── results/
    ├── logs/          — JSON results and training histories
    ├── models/        — Saved .pt model checkpoints
    └── visualizations/— Confusion matrices and training curve plots
```
