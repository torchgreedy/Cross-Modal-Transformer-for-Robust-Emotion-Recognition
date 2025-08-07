import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureImportanceModule(nn.Module):
    """
    Feature importance gating mechanism for adaptive feature selection
    
    Uses a gating mechanism to learn which features are most important
    for the task, allowing the model to focus on relevant information.
    """
    
    def __init__(self, feature_dimension, reduction_ratio=16, activation='sigmoid'):
        super().__init__()
        
        self.feature_dimension = feature_dimension
        self.activation_type = activation
        
        # Feature importance gate
        self.gate = nn.Sequential(
            nn.Linear(feature_dimension, feature_dimension // reduction_ratio),
            nn.GELU(),
            nn.Linear(feature_dimension // reduction_ratio, feature_dimension)
        )
        
        # Activation function for gating
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = lambda x: F.softmax(x, dim=-1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gating weights"""
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Apply feature importance gating
        
        Args:
            x: Input features (B, T, feature_dimension) or (B, feature_dimension)
            
        Returns:
            Tuple of (gated_features, importance_weights)
        """
        # Compute importance weights
        if x.dim() == 3:  # Sequence data (B, T, F)
            # Global average pooling for gate computation
            pooled = x.mean(dim=1)  # (B, F)
            importance = self.activation(self.gate(pooled))  # (B, F)
            importance = importance.unsqueeze(1)  # (B, 1, F)
        else:  # Non-sequence data (B, F)
            importance =
