import torch
import torch.nn as nn
from src.modules.projections import PositionalEncoding
from src.modules.attention import CrossModalAttention, TransformerEncoderLayer
from src.modules.grl import DomainAdaptationLayer
from src.modules.fusion import FeatureImportanceModule

class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer for emotion recognition using EEG and eye-tracking data.
    
    This model processes multiple modalities through:
    1. Modality-specific projections
    2. Cross-modal attention
    3. Self-attention
    4. Gated fusion for final prediction
    """
    def __init__(self, eeg_features=310, eye_features=33, d_model=256, n_heads=8,
                 n_encoder_layers=4, n_classes=5, n_subjects=16, dropout=0.1):
        super().__init__()

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
        Forward pass through the Cross-Modal Transformer.
        
        Args:
            eeg: Tensor of shape [batch_size, seq_len, eeg_features]
            eye: Tensor of shape [batch_size, seq_len, eye_features]
            eeg_mask: Mask for EEG features (1 for valid, 0 for padding)
            eye_mask: Mask for eye tracking features
            groups: Subject IDs for domain adaptation
            alpha: GRL gradient scaling parameter
            
        Returns:
            Dictionary containing:
                - 'emotion_logits': Emotion classification logits
                - 'domain_logits': Subject classification logits
                - 'eeg_importance': EEG feature importance weights
                - 'eye_importance': Eye feature importance weights
                - 'cross_modal_attn': Cross-modal attention weights
        """
        # Input projections
        eeg_projected = self.eeg_projection(eeg)  # (B, T, d_model)
        eye_projected = self.eye_projection(eye)  # (B, T, d_model)

        # Positional encoding
        eeg_pos = self.pos_encoding(eeg_projected)
        eye_pos = self.pos_encoding(eye_projected)

        # Feature importance
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
        if eeg_mask is not None:
            eeg_mask_expanded = eeg_mask.unsqueeze(-1).expand_as(eeg_encoded)
            eeg_pooled = (eeg_encoded * eeg_mask_expanded).sum(dim=1) / (eeg_mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            eeg_pooled = eeg_encoded.mean(dim=1)

        if eye_mask is not None:
            eye_mask_expanded = eye_mask.unsqueeze(-1).expand_as(eye_encoded)
            eye_pooled = (eye_encoded * eye_mask_expanded).sum(dim=1) / (eye_mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            eye_pooled = eye_encoded.mean(dim=1)

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
