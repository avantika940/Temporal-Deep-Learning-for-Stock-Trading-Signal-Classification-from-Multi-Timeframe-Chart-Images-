"""
Multiple BiLSTM model architectures for stock signal classification

This module contains 5 advanced BiLSTM variants designed to improve upon
Step 4's best result (61.21% from Temporal Transformer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import math


class DeepBiLSTM(nn.Module):
    """
    Deep stacked BiLSTM baseline model
    
    Architecture:
    - Frozen ViT feature extractor
    - 3 stacked BiLSTM layers (256, 256, 128)
    - Dropout between layers
    - Uses last hidden state for classification
    
    Expected improvement: +0-2% over Step 4
    """
    
    def __init__(self, config):
        super(DeepBiLSTM, self).__init__()
        
        self.config = config
        
        # Load pre-trained ViT as frozen feature extractor
        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # 3-layer stacked BiLSTM
        self.lstm1 = nn.LSTM(
            input_size=config.VIT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=config.HIDDEN_DIM * 2,
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm3 = nn.LSTM(
            input_size=config.HIDDEN_DIM * 2,
            hidden_size=config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(config.LSTM_DROPOUT)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, C, H, W]
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len, C, H, W = x.shape
        
        # Extract ViT features
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.no_grad():
            vit_out = self.vit(x)
            features = vit_out.last_hidden_state[:, 0, :]  # CLS token
        features = features.view(batch_size, seq_len, -1)
        
        # Pass through 3 BiLSTM layers
        x, _ = self.lstm1(features)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        x, _ = self.lstm3(x)
        
        # Use last output
        x = x[:, -1, :]
        
        # Classify
        logits = self.classifier(x)
        return logits


class AttentionBiLSTM(nn.Module):
    """
    BiLSTM with self-attention mechanism
    
    Architecture:
    - Frozen ViT feature extractor
    - 2-layer BiLSTM
    - Self-attention over LSTM outputs
    - Attention-weighted pooling
    
    Expected improvement: +1-3% over Step 4
    """
    
    def __init__(self, config):
        super(AttentionBiLSTM, self).__init__()
        
        self.config = config
        
        # ViT feature extractor
        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=config.VIT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            dropout=config.LSTM_DROPOUT if 2 > 1 else 0,
            bidirectional=True
        )
        
        # Self-attention mechanism
        self.attention = SelfAttention(config.HIDDEN_DIM * 2, config.ATTENTION_DROPOUT)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        
        # Extract ViT features
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.no_grad():
            vit_out = self.vit(x)
            features = vit_out.last_hidden_state[:, 0, :]
        features = features.view(batch_size, seq_len, -1)
        
        # BiLSTM processing
        lstm_out, _ = self.bilstm(features)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Classify
        logits = self.classifier(context)
        return logits


class ResidualBiLSTM(nn.Module):
    """
    BiLSTM with residual connections
    
    Architecture:
    - Frozen ViT feature extractor
    - 3 BiLSTM layers with skip connections
    - Layer normalization
    - Better gradient flow for deeper networks
    
    Expected improvement: +1-3% over Step 4
    """
    
    def __init__(self, config):
        super(ResidualBiLSTM, self).__init__()
        
        self.config = config
        
        # ViT feature extractor
        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Project ViT features to HIDDEN_DIM * 2 (for residual connection)
        self.input_projection = nn.Linear(config.VIT_DIM, config.HIDDEN_DIM * 2)
        
        # Residual BiLSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=config.HIDDEN_DIM * 2,
                hidden_size=config.HIDDEN_DIM,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) for _ in range(3)
        ])
        
        # Layer normalization
        if config.USE_LAYER_NORM:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(config.HIDDEN_DIM * 2) for _ in range(3)
            ])
        else:
            self.layer_norms = None
        
        self.dropout = nn.Dropout(config.RESIDUAL_DROPOUT)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        
        # Extract ViT features
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.no_grad():
            vit_out = self.vit(x)
            features = vit_out.last_hidden_state[:, 0, :]
        features = features.view(batch_size, seq_len, -1)
        
        # Project to correct dimension
        x = self.input_projection(features)
        
        # Pass through residual BiLSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            residual = x
            x, _ = lstm(x)
            x = self.dropout(x)
            
            # Add residual connection
            x = x + residual
            
            # Layer normalization
            if self.layer_norms is not None:
                # Apply layer norm across feature dimension
                x = x.transpose(1, 2)  # [batch, features, seq]
                x = self.layer_norms[i](x.transpose(1, 2))  # Back to [batch, seq, features]
        
        # Use last output
        x = x[:, -1, :]
        
        # Classify
        logits = self.classifier(x)
        return logits


class HybridBiLSTM(nn.Module):
    """
    Hybrid model combining attention and residual connections
    
    Architecture:
    - Frozen ViT feature extractor
    - 3 residual BiLSTM layers
    - Multi-head self-attention
    - Gated fusion mechanism
    - Most sophisticated model
    
    Expected improvement: +2-5% over Step 4 ⭐ BEST
    """
    
    def __init__(self, config):
        super(HybridBiLSTM, self).__init__()
        
        self.config = config
        
        # ViT feature extractor
        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Input projection
        self.input_projection = nn.Linear(config.VIT_DIM, config.HIDDEN_DIM * 2)
        
        # Residual BiLSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=config.HIDDEN_DIM * 2,
                hidden_size=config.HIDDEN_DIM,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) for _ in range(3)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.HIDDEN_DIM * 2) for _ in range(3)
        ])
        
        self.dropout = nn.Dropout(config.LSTM_DROPOUT)
        
        # Multi-head self-attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM * 2,
            num_heads=config.ATTENTION_HEADS,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True
        )
        
        # Gated fusion (combines LSTM output and attention output)
        self.gate = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 4, config.HIDDEN_DIM * 2),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        
        # Extract ViT features
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.no_grad():
            vit_out = self.vit(x)
            features = vit_out.last_hidden_state[:, 0, :]
        features = features.view(batch_size, seq_len, -1)
        
        # Project features
        x = self.input_projection(features)
        
        # Pass through residual BiLSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            residual = x
            x, _ = lstm(x)
            x = self.dropout(x)
            x = x + residual
            x = x.transpose(1, 2)
            x = self.layer_norms[i](x.transpose(1, 2))
        
        lstm_output = x
        
        # Apply multi-head attention
        attention_output, _ = self.multihead_attention(x, x, x)
        
        # Gated fusion
        combined = torch.cat([lstm_output, attention_output], dim=-1)
        gate_values = self.gate(combined)
        
        fused = gate_values * lstm_output + (1 - gate_values) * attention_output
        
        # Mean pooling over sequence
        pooled = fused.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        return logits


class PyramidalBiLSTM(nn.Module):
    """
    Pyramidal BiLSTM with hierarchical processing
    
    Architecture:
    - Frozen ViT feature extractor
    - 3 BiLSTM layers with decreasing hidden sizes (512→256→128)
    - Multi-scale temporal pyramid pooling
    - Captures both fine-grained and high-level patterns
    
    Expected improvement: +1-4% over Step 4
    """
    
    def __init__(self, config):
        super(PyramidalBiLSTM, self).__init__()
        
        self.config = config
        
        # ViT feature extractor
        self.vit = ViTModel.from_pretrained(config.VIT_MODEL)
        if config.FREEZE_VIT:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Pyramidal BiLSTM layers (decreasing sizes)
        self.lstm1 = nn.LSTM(
            input_size=config.VIT_DIM,
            hidden_size=config.HIDDEN_DIM * 2,  # 512
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=config.HIDDEN_DIM * 4,  # 512*2 (bidirectional)
            hidden_size=config.HIDDEN_DIM,  # 256
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm3 = nn.LSTM(
            input_size=config.HIDDEN_DIM * 2,  # 256*2
            hidden_size=config.HIDDEN_DIM // 2,  # 128
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(config.LSTM_DROPOUT)
        
        # Temporal pyramid pooling
        self.pyramid_pooling = TemporalPyramidPooling(
            config.HIDDEN_DIM,  # 128*2 = 256 from last BiLSTM
            [1, 2, 5]  # Pool over 1, 2, and 5 frames (fits seq_len=10 evenly)
        )
        
        # Calculate pooled feature size: 256 * (1 + 2 + 5) = 2048
        pooled_dim = config.HIDDEN_DIM * (1 + 2 + 5)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        
        # Extract ViT features
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.no_grad():
            vit_out = self.vit(x)
            features = vit_out.last_hidden_state[:, 0, :]
        features = features.view(batch_size, seq_len, -1)
        
        # Pyramidal BiLSTM
        x, _ = self.lstm1(features)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        x, _ = self.lstm3(x)
        
        # Temporal pyramid pooling
        x = self.pyramid_pooling(x)
        
        # Classify
        logits = self.classifier(x)
        return logits


# ============= Helper Modules =============

class SelfAttention(nn.Module):
    """Self-attention mechanism for sequence modeling"""
    
    def __init__(self, hidden_dim, dropout=0.2):
        super(SelfAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: [batch, seq_len, hidden_dim]
        Returns:
            context: [batch, hidden_dim]
            attention_weights: [batch, seq_len]
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)  # [batch, seq_len]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len]
        
        # Apply attention weights
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, seq_len]
            lstm_output  # [batch, seq_len, hidden_dim]
        ).squeeze(1)  # [batch, hidden_dim]
        
        context = self.dropout(context)
        
        return context, attention_weights


class TemporalPyramidPooling(nn.Module):
    """Temporal pyramid pooling for multi-scale features"""
    
    def __init__(self, hidden_dim, pool_sizes=[1, 2, 3]):
        super(TemporalPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            pooled: [batch, hidden_dim * sum(pool_sizes)]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        pooled_features = []
        
        for pool_size in self.pool_sizes:
            if pool_size == 1:
                # Global average pooling
                pooled = x.mean(dim=1)  # [batch, hidden_dim]
            else:
                # Average pool over chunks
                chunk_size = seq_len // pool_size
                chunks = []
                for i in range(pool_size):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < pool_size - 1 else seq_len
                    chunk = x[:, start_idx:end_idx, :].mean(dim=1)
                    chunks.append(chunk)
                pooled = torch.cat(chunks, dim=-1)  # [batch, hidden_dim * pool_size]
            
            pooled_features.append(pooled)
        
        return torch.cat(pooled_features, dim=-1)


# ============= Model Factory =============

def create_model(config):
    """
    Factory function to create model based on config
    
    Args:
        config: Configuration object
        
    Returns:
        Model instance
    """
    models = {
        'deep': DeepBiLSTM,
        'attention': AttentionBiLSTM,
        'residual': ResidualBiLSTM,
        'hybrid': HybridBiLSTM,
        'pyramidal': PyramidalBiLSTM
    }
    
    model_class = models.get(config.MODEL_TYPE.lower())
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}. "
                        f"Choose from: {list(models.keys())}")
    
    model = model_class(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {config.MODEL_TYPE.upper()}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import Config
    
    for model_type in ['deep', 'attention', 'residual', 'hybrid', 'pyramidal']:
        print(f"\n{'='*80}")
        print(f"Testing {model_type.upper()} model")
        print('='*80)
        
        config = Config(MODEL_TYPE=model_type, SEQUENCE_LENGTH=10)
        model = create_model(config)
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 10, 3, 224, 224)
        
        try:
            output = model(dummy_input)
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected output shape: [{batch_size}, {config.NUM_CLASSES}]")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
