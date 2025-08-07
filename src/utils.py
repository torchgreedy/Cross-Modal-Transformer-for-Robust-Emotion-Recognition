import random
import numpy as np
import torch

def normalize_features(features, mask):
    """Normalize features using only non-padded values."""
    N, T, F = features.shape
    features_reshaped = features.reshape(-1, F)
    mask_reshaped = mask.reshape(-1)
    non_padded = features_reshaped[mask_reshaped.astype(bool)]
    mean = non_padded.mean(axis=0)
    std = non_padded.std(axis=0)
    features_norm = (features - mean) / (std + 1e-8)
    return features_norm, mean, std

def apply_normalization(features, mean, std):
    """Apply pre-computed normalization parameters to features."""
    return (features - mean) / (std + 1e-8)

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
