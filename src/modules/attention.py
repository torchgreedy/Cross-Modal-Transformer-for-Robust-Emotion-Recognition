import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for EEG-Eye movement interaction
    
    Implements bidirectional attention where:
    - EEG queries Eye: What eye movements are relevant to brain states?
    - Eye queries EEG: What brain states are relevant to eye movements?
    """
    
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Separate projections for each modality
        self.eeg_q = nn.Linear(d_model, d_model)
        self.eeg_k = nn.Linear(d_model, d_model)
        self.eeg_v = nn.Linear(d_model, d_model)
        
        self.eye_q = nn.Linear(d_model, d_model)
        self.eye_k = nn.Linear(d_model, d_model)
        self.eye_v = nn.Linear(d_model, d_model)
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights using Xavier uniform"""
        for module in [self.eeg_q, self.eeg_k, self.eeg_v, 
                      self.eye_q, self.eye_k, self.eye_v, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, eeg_features, eye_features, eeg_mask=None, eye_mask=None):
        """
        Compute cross-modal attention between EEG and eye movement features
        
        Args:
            eeg_features: EEG features (B, T, d_model)
            eye_features: Eye movement features (B, T, d_model)
            eeg_mask: EEG sequence mask (B, T)
            eye_mask: Eye movement sequence mask (B, T)
            
        Returns:
            Tuple of (eeg_fused, eye_fused, attention_weights)
        """
        batch_size, seq_len, _ = eeg_features.shape
        
        # EEG queries Eye: What eye movements are relevant to brain states?
        eeg_q = self._reshape_for_attention(self.eeg_q(eeg_features))
        eye_k = self._reshape_for_attention(self.eye_k(eye_features))
        eye_v = self._reshape_for_attention(self.eye_v(eye_features))
        
        # Compute EEG-Eye attention
        eeg_eye_scores = torch.matmul(eeg_q, eye_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if eye_mask is not None:
            eye_mask_expanded = self._expand_mask(eye_mask, seq_len)
            eeg_eye_scores = eeg_eye_scores.masked_fill(eye_mask_expanded == 0, -1e9)
        
        eeg_eye_attn = F.softmax(eeg_eye_scores, dim=-1)
        eeg_eye_attn = self.dropout(eeg_eye_attn)
        eeg_attended = torch.matmul(eeg_eye_attn, eye_v)
        
        # Eye queries EEG: What brain states are relevant to eye movements?
        eye_q = self._reshape_for_attention(self.eye_q(eye_features))
        eeg_k = self._reshape_for_attention(self.eeg_k(eeg_features))
        eeg_v = self._reshape_for_attention(self.eeg_v(eeg_features))
        
        # Compute Eye-EEG attention
        eye_eeg_scores = torch.matmul(eye_q, eeg_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if eeg_mask is not None:
            eeg_mask_expanded = self._expand_mask(eeg_mask, seq_len)
            eye_eeg_scores = eye_eeg_scores.masked_fill(eeg_mask_expanded == 0, -1e9)
        
        eye_eeg_attn = F.softmax(eye_eeg_scores, dim=-1)
        eye_eeg_attn = self.dropout(eye_eeg_attn)
        eye_attended = torch.matmul(eye_eeg_attn, eeg_v)
        
        # Reshape back to original dimensions
        eeg_attended = self._reshape_from_attention(eeg_attended, batch_size, seq_len)
        eye_attended = self._reshape_from_attention(eye_attended, batch_size, seq_len)
        
        # Apply output projection and residual connections
        eeg_fused = eeg_features + self.output_proj(eeg_attended)
        eye_fused = eye_features + self.output_proj(eye_attended)
        
        # Return attention weights for interpretability (averaged over heads)
        attention_weights = {
            'eeg_eye_attn': eeg_eye_attn.mean(dim=1),  # (B, T, T)
            'eye_eeg_attn': eye_eeg_attn.mean(dim=1)   # (B, T, T)
        }
        
        return eeg_fused, eye_fused, attention_weights
    
    def _reshape_for_attention(self, x):
        """Reshape tensor for multi-head attention"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    
    def _reshape_from_attention(self, x, batch_size, seq_len):
        """Reshape tensor back from multi-head attention"""
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def _expand_mask(self, mask, seq_len):
        """Expand mask for multi-head attention"""
        return mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, seq_len, -1)


class TransformerEncoderLayer(nn.Module):
    """
    Standard transformer encoder layer with self-attention and feed-forward
    """
    
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass through transformer encoder layer
        
        Args:
            x: Input features (B, T, d_model)
            mask: Sequence mask (B, T)
            
        Returns:
            Encoded features (B, T, d_model)
        """
        # Self-attention with residual connection
        key_padding_mask = None
        if mask is not None:
            # Create key padding mask (True for positions to ignore)
            key_padding_mask = mask == 0  # (B, T)
        
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, seq_len, -1)
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(context)
        
        return output, attention_weights.mean(dim=1)  # Return averaged attention weights
