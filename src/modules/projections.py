import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models
    
    Adds position information to input embeddings using sine and cosine functions
    of different frequencies.
    """
    
    def __init__(self, d_model=256, max_len=74):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        # Create division term for different frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            
        Returns:
            Position-encoded embeddings
        """
        # Select appropriate number of positions
        positions = self.pe[:, :x.size(1)]
        return x + positions


class InputProjection(nn.Module):
    """
    Projects input features to model dimension with optional normalization
    """
    
    def __init__(self, input_dim, d_model, dropout=0.1, use_layer_norm=True):
        super().__init__()
        
        self.projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
    
    def forward(self, x):
        """
        Project input features to model dimension
        
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            
        Returns:
            Projected features (batch_size, seq_len, d_model)
        """
        x = self.projection(x)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
            
        x = self.dropout(x)
        return x
