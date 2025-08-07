import torch
import torch.nn as nn
import numpy as np


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial domain adaptation
    
    During forward pass, acts as identity function
    During backward pass, reverses gradients and scales by alpha
    """
    
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        """Forward pass - identity function"""
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - reverse and scale gradients"""
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x, alpha=1.0):
    """Convenience function for applying gradient reversal"""
    return GradientReversalLayer.apply(x, alpha)


class DomainAdaptationLayer(nn.Module):
    """
    Domain adaptation layer with gradient reversal and subject-specific normalization
    
    Combines adversarial training (via gradient reversal) with subject-specific
    batch normalization to improve cross-subject generalization.
    """
    
    def __init__(self, d_model=512, n_subjects=16, dropout=0.1):
        super().__init__()
        
        self.n_subjects = n_subjects
        
        # Domain classifier for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_subjects)
        )
        
        # Subject-specific normalization layers
        self.subject_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_subjects)
        ])
        
        # Default normalization for unknown subjects
        self.default_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize domain classifier weights"""
        for module in self.domain_classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, groups=None, alpha=1.0):
        """
        Forward pass with domain adaptation
        
        Args:
            x: Input features (B, d_model)
            groups: Subject group labels (B,) - can be None during inference
            alpha: Gradient reversal strength
            
        Returns:
            Tuple of (normalized_features, domain_logits)
        """
        # Gradient reversal for adversarial training
        domain_features = gradient_reversal(x, alpha)
        domain_logits = self.domain_classifier(domain_features)
        
        # Subject-specific normalization
        normalized_x = self._apply_subject_normalization(x, groups)
        
        return normalized_x, domain_logits
    
    def _apply_subject_normalization(self, x, groups):
        """Apply subject-specific layer normalization"""
        if groups is None:
            # During inference without group labels, use default normalization
            return self.default_norm(x)
        
        # Initialize output tensor
        normalized_x = torch.zeros_like(x)
        
        # Apply subject-specific normalization
        for subject_id in range(self.n_subjects):
            mask = (groups == subject_id)
            if mask.any():
                normalized_x[mask] = self.subject_norms[subject_id](x[mask])
        
        # Handle any subjects not in range [0, n_subjects-1]
        unhandled_mask = (groups >= self.n_subjects) | (groups < 0)
        if unhandled_mask.any():
            normalized_x[unhandled_mask] = self.default_norm(x[unhandled_mask])
        
        return normalized_x
    
    def get_domain_predictions(self, x, alpha=0.0):
        """Get domain predictions without gradient reversal (for evaluation)"""
        domain_features = gradient_reversal(x, alpha)
        return self.domain_classifier(domain_features)


class AdversarialLoss(nn.Module):
    """
    Combined loss function for adversarial domain adaptation
    """
    
    def __init__(self, emotion_weight=1.0, domain_weight=0.1):
        super().__init__()
        
        self.emotion_weight = emotion_weight
        self.domain_weight = domain_weight
        
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
    
    def forward(self, emotion_logits, domain_logits, emotion_labels, domain_labels):
        """
        Compute combined adversarial loss
        
        Args:
            emotion_logits: Predicted emotion logits
            domain_logits: Predicted domain logits  
            emotion_labels: True emotion labels
            domain_labels: True domain (subject) labels
            
        Returns:
            Dictionary containing individual and total losses
        """
        emotion_loss = self.emotion_criterion(emotion_logits, emotion_labels)
        domain_loss = self.domain_criterion(domain_logits, domain_labels)
        
        # Total loss: minimize emotion loss, maximize domain confusion
        total_loss = self.emotion_weight * emotion_loss + self.domain_weight * domain_loss
        
        return {
            'total_loss': total_loss,
            'emotion_loss': emotion_loss,
            'domain_loss': domain_loss
        }


def compute_gradient_reversal_alpha(epoch, total_epochs, max_alpha=1.0):
    """
    Compute gradient reversal strength that increases over training
    
    Uses the schedule from the original DANN paper:
    alpha = 2 / (1 + exp(-10 * p)) - 1
    where p is the training progress [0, 1]
    """
    progress = epoch / total_epochs
    alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
    return alpha * max_alpha


class DomainConfusionLoss(nn.Module):
    """
    Alternative domain confusion loss that directly maximizes entropy
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, domain_logits):
        """
        Compute domain confusion loss by maximizing prediction entropy
        
        Args:
            domain_logits: Domain classifier predictions
            
        Returns:
            Domain confusion loss (negative entropy)
        """
        domain_probs = torch.softmax(domain_logits, dim=-1)
        entropy = -torch.sum(domain_probs * torch.log(domain_probs + 1e-8), dim=-1)
        return -entropy.mean()  # Maximize entropy = minimize negative entropy
