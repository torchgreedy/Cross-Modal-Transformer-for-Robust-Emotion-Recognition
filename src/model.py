import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.projections import PositionalEncoding
from .modules.attention import CrossModalAttention, TransformerEncoderLayer
from .modules.grl import DomainAdaptationLayer
from .modules.fusion import FeatureImportanceModule


class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer for EEG-Eye Movement Emotion Recognition
    
    This model combines EEG and eye movement data through cross-modal attention
    and includes domain adaptation for subject-independent emotion recognition.
    """
    
    def __init__(self, eeg_features=310, eye_features=33, d_model=256, n_heads=8,
                 n_encoder_layers=4, n_classes=5, n_subjects=16, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_subjects = n_subjects
        
        # Input projections
        self.eeg_projection = nn.Linear(eeg_features, d_model)
        self.eye_projection = nn.Linear(eye_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Feature importance modules
        self.eeg_importance = FeatureImportanceModule(d_model)
        self.eye_importance = FeatureImportanceModule(d_model)
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(d_model, n_heads, dropout)
        
        # Transformer encoders
        self.eeg_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.eye_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Domain adaptation
        self.domain_adaptation = DomainAdaptationLayer(d_model*2, n_subjects)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model*2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, eeg, eye, eeg_mask=None, eye_mask=None, groups=None, alpha=1.0):
        """
        Forward pass of the cross-modal transformer
        
        Args:
            eeg: EEG features (B, T, eeg_features)
            eye: Eye movement features (B, T, eye_features)
            eeg_mask: EEG sequence mask (B, T)
            eye_mask: Eye movement sequence mask (B, T)
            groups: Subject groups for domain adaptation (B,)
            alpha: Gradient reversal strength
            
        Returns:
            Dictionary containing emotion logits, domain logits, and attention weights
        """
        # Input projections
        eeg_projected = self.eeg_projection(eeg)  # (B, T, d_model)
        eye_projected = self.eye_projection(eye)  # (B, T, d_model)
        
        # Positional encoding
        eeg_pos = self.pos_encoding(eeg_projected)
        eye_pos = self.pos_encoding(eye_projected)
        
        # Feature importance gating
        eeg_weighted, eeg_importance = self.eeg_importance(eeg_pos)
        eye_weighted, eye_importance = self.eye_importance(eye_pos)
        
        # Cross-modal attention
        eeg_fused, eye_fused, cross_modal_attn = self.cross_modal_attention(
            eeg_weighted, eye_weighted, eeg_mask, eye_mask
        )
        
        # Self-attention encoders
        eeg_encoded = eeg_fused
        for layer in self.eeg_encoder:
            eeg_encoded = layer(eeg_encoded, eeg_mask)
        
        eye_encoded = eye_fused
        for layer in self.eye_encoder:
            eye_encoded = layer(eye_encoded, eye_mask)
        
        # Global pooling with masking
        eeg_pooled = self._masked_global_pool(eeg_encoded, eeg_mask)
        eye_pooled = self._masked_global_pool(eye_encoded, eye_mask)
        
        # Feature fusion
        fused_features = torch.cat([eeg_pooled, eye_pooled], dim=-1)  # (B, d_model*2)
        
        # Domain adaptation
        domain_adapted, domain_logits = self.domain_adaptation(fused_features, groups, alpha)
        
        # Classification
        emotion_logits = self.classifier(domain_adapted)
        
        return {
            'emotion_logits': emotion_logits,
            'domain_logits': domain_logits,
            'eeg_importance': eeg_importance,
            'eye_importance': eye_importance,
            'cross_modal_attn': cross_modal_attn
        }
    
    def _masked_global_pool(self, features, mask):
        """Apply global average pooling with sequence masking"""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(features)
            pooled = (features * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            pooled = features.mean(dim=1)
        return pooled
    
    def get_attention_weights(self, eeg, eye, eeg_mask=None, eye_mask=None):
        """Extract attention weights for interpretability"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(eeg, eye, eeg_mask, eye_mask, alpha=0.0)
            return {
                'eeg_importance': outputs['eeg_importance'],
                'eye_importance': outputs['eye_importance'],
                'cross_modal_attn': outputs['cross_modal_attn']
            }
