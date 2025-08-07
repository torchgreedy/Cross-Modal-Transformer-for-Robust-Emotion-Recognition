# Feature Importance Module (FIM)

## Purpose

The Feature Importance Module is designed to learn modality-specific importance weights that indicate how relevant each modality is for emotion recognition, adaptively across different samples, timesteps, and feature dimensions. This helps:

- Emphasize stronger signals (e.g., if EEG is more informative for a sample, weight it more)
- Reduce reliance on noisy or less relevant modalities

## Architectural Role

Assume that after linear projection and positional encoding, we obtain:

- Z_eeg ∈ ℝ^(B × T × d)  
- Z_eye ∈ ℝ^(B × T × d)

These represent the EEG and eye-tracking features, respectively, where:

- B: Batch size  
- T: Sequence length (timesteps)  
- d: Feature dimension after projection

These modality-specific features are processed independently at first. Before fusion or cross-modal attention, we apply the Feature Importance Module to reweight each feature tensor.

## Mathematical Formulation

### 1. Modality-Specific Importance Estimation

We use a small feedforward network to compute a scalar importance score for each modality at each timestep.

Let:

- W_f ∈ ℝ^(d × 1),  
- b_f ∈ ℝ

Then, for each timestep t ∈ [1, T], compute:

- alpha_eeg[t] = sigmoid(Z_eeg[t] · W_f + b_f) ∈ ℝ  
- alpha_eye[t] = sigmoid(Z_eye[t] · W_f + b_f) ∈ ℝ

Where `sigmoid(·)` denotes the sigmoid activation function to squash values to the range [0, 1].

### 2. Reweighting the Modalities

We then reweight each modality's features using the learned importance scores:

- Z_eeg_tilde[t] = alpha_eeg[t] * Z_eeg[t]  
- Z_eye_tilde[t] = alpha_eye[t] * Z_eye[t]

These rescaled embeddings `Z_eeg_tilde` and `Z_eye_tilde` are passed to the subsequent fusion or attention modules in the model.
