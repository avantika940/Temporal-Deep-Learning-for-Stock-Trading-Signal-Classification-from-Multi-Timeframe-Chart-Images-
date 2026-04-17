# Technical Documentation

## Architecture Deep Dive

### Feature Engineering Pipeline

#### 1. Visual Feature Extraction
```
Input: MTF Image (224x224x3) 
  ↓
ViT Preprocessing: Patch extraction (16x16 patches)
  ↓
ViT Forward Pass: Self-attention mechanism
  ↓
CLS Token Extraction: 768-dimensional embedding
  ↓
Output: Visual features (768-dim)
```

#### 2. Technical Indicator Calculation
```python
def calculate_technical_indicators(label_sequence):
    """
    Calculate 5 technical indicators from BUY/HOLD/SELL label sequence
    
    Args:
        label_sequence: List of labels mapped as {BUY: +1, HOLD: 0, SELL: -1}
    
    Returns:
        indicators: 5-dimensional array of technical indicators
    """
    # 1. RSI-proxy
    gains = [max(0, x) for x in np.diff(label_sequence)]
    losses = [abs(min(0, x)) for x in np.diff(label_sequence)]
    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 1
    rsi_proxy = avg_gain / (avg_gain + avg_loss)
    
    # 2. MACD-proxy
    ema_fast = exponential_moving_average(label_sequence, span=3)
    ema_slow = exponential_moving_average(label_sequence, span=6)
    macd_proxy = (ema_fast - ema_slow) / 2.0  # Normalized to [-1, 1]
    
    # 3. MACD-signal
    macd_signal = exponential_moving_average([macd_proxy], span=3)[0]
    
    # 4. Momentum
    momentum = (label_sequence[-1] - label_sequence[0]) / len(label_sequence)
    
    # 5. Volume-weighted signal (simplified)
    volume_weight = 1.0  # Constant for this implementation
    volume_weighted = np.mean(label_sequence) * volume_weight
    
    return np.array([rsi_proxy, macd_proxy, macd_signal, momentum, volume_weighted])
```

### Model Architecture Details

#### HybridBiGRU Architecture
```python
class HybridBiGRU(nn.Module):
    def __init__(self, input_dim=773, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        gru_out, _ = self.bigru(x)  # (batch_size, seq_len, hidden_dim*2)
        last_output = gru_out[:, -1, :]  # Take last timestamp
        dropped = self.dropout(last_output)
        logits = self.classifier(dropped)
        return logits
```

#### HybridBiLSTM Architecture
```python
class HybridBiLSTM(nn.Module):
    def __init__(self, input_dim=773, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.bilstm(x)  # (batch_size, seq_len, hidden_dim*2)
        last_output = lstm_out[:, -1, :]  # Take last timestamp
        dropped = self.dropout(last_output)
        logits = self.classifier(dropped)
        return logits
```

### Data Processing Pipeline

#### 1. Chronological Data Splitting
```python
def chronological_split(sequences, train_ratio=0.64, val_ratio=0.16):
    """
    Split sequences chronologically to prevent data leakage
    
    Args:
        sequences: List of sequences sorted by timestamp
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        train_seq, val_seq, test_seq: Chronologically split sequences
    """
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return sequences[:train_end], sequences[train_end:val_end], sequences[val_end:]
```

#### 2. Sequence Generation
```python
def create_sequences(images, labels, seq_len=10, stride=1):
    """
    Create overlapping sequences from time-series data
    
    Args:
        images: List of image paths sorted by timestamp
        labels: Corresponding labels
        seq_len: Length of each sequence
        stride: Step size for sliding window
        
    Returns:
        sequences: List of (image_sequence, label) tuples
    """
    sequences = []
    for i in range(0, len(images) - seq_len + 1, stride):
        img_seq = images[i:i + seq_len]
        target_label = labels[i + seq_len - 1]  # Predict last label in sequence
        sequences.append((img_seq, target_label))
    
    return sequences
```

### Training Configuration

#### Hyperparameters
```python
TRAINING_CONFIG = {
    'epochs': 80,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'optimizer': 'Adam',
    'scheduler': 'CosineAnnealingLR',
    'eta_min': 1e-6,
    'weight_decay': 1e-5,
    'dropout': 0.3,
    'hidden_dim': 128,
    'seq_len': 10,
    'stride': 1
}
```

#### Class Weighting Strategy
```python
def calculate_class_weights(labels):
    """
    Calculate inverse frequency weights for imbalanced classes
    
    Args:
        labels: List of class labels
        
    Returns:
        weights: Dictionary of class weights
    """
    from collections import Counter
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = {}
    for class_label, count in class_counts.items():
        weights[class_label] = total_samples / (num_classes * count)
    
    return weights
```

### Ensemble Methods Implementation

#### Soft Voting Ensemble
```python
def soft_voting_ensemble(predictions_list):
    """
    Combine predictions using averaged probabilities
    
    Args:
        predictions_list: List of prediction arrays from different models
        
    Returns:
        ensemble_predictions: Combined predictions
    """
    avg_predictions = np.mean(predictions_list, axis=0)
    return np.argmax(avg_predictions, axis=1)
```

#### Weighted Voting Ensemble
```python
def weighted_voting_ensemble(predictions_list, weights):
    """
    Combine predictions using weighted probabilities
    
    Args:
        predictions_list: List of prediction arrays
        weights: List of weights for each model
        
    Returns:
        ensemble_predictions: Weighted combined predictions
    """
    weighted_sum = np.zeros_like(predictions_list[0])
    for i, pred in enumerate(predictions_list):
        weighted_sum += pred * weights[i]
    
    return np.argmax(weighted_sum, axis=1)
```

### Performance Metrics

#### Custom Metrics Implementation
```python
def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix, classification_report
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'support': support.tolist()
    }
    
    return metrics
```

### Memory Optimization

#### Efficient Data Loading
```python
class EfficientDataLoader:
    """
    Memory-efficient data loader for large datasets
    """
    def __init__(self, sequences, batch_size=32, shuffle=False):
        self.sequences = sequences
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self._load_batch(batch_indices)
    
    def _load_batch(self, indices):
        """Load and process a batch of sequences"""
        batch_sequences = []
        batch_labels = []
        
        for idx in indices:
            seq, label = self.sequences[idx]
            # Load images on-demand to save memory
            processed_seq = self._process_sequence(seq)
            batch_sequences.append(processed_seq)
            batch_labels.append(label)
        
        return np.array(batch_sequences), np.array(batch_labels)
```

### Debugging and Monitoring

#### Training Progress Monitoring
```python
class TrainingMonitor:
    """
    Monitor training progress and detect issues
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Log metrics for each epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # Check for potential issues
        self._check_overfitting(epoch)
        self._check_underfitting(epoch)
    
    def _check_overfitting(self, epoch):
        """Detect overfitting patterns"""
        if epoch > 10:
            recent_val_loss = np.mean(self.val_losses[-5:])
            early_val_loss = np.mean(self.val_losses[5:10])
            
            if recent_val_loss > early_val_loss * 1.1:
                print(f"Warning: Possible overfitting detected at epoch {epoch}")
    
    def _check_underfitting(self, epoch):
        """Detect underfitting patterns"""
        if epoch > 20:
            recent_train_acc = np.mean(self.train_accs[-5:])
            if recent_train_acc < 0.7:
                print(f"Warning: Possible underfitting detected at epoch {epoch}")
```

### Error Handling and Recovery

#### Robust Training Loop
```python
def robust_training_loop(model, train_loader, val_loader, epochs=80):
    """
    Training loop with error handling and recovery
    """
    checkpoint_freq = 10
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        try:
            # Training phase
            train_loss, train_acc = train_epoch(model, train_loader)
            
            # Validation phase
            val_loss, val_acc = validate_epoch(model, val_loader)
            
            # Save checkpoint
            if epoch % checkpoint_freq == 0:
                save_checkpoint(model, epoch, val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_best_model(model)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        except Exception as e:
            print(f"Error at epoch {epoch}: {str(e)}")
            # Attempt to recover from checkpoint
            if epoch > checkpoint_freq:
                model = load_latest_checkpoint()
                print("Recovered from checkpoint")
            else:
                raise e
    
    return model
```

This technical documentation provides detailed insights into the implementation specifics, optimization strategies, and best practices used in the Multi-Stock MTF Classification system.
