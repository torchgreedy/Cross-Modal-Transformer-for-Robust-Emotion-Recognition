# Global Pooling and Fusion

## Global Pooling

After modality-specific Transformer encoders, we extract a fixed-size representation for each modality by applying global average pooling across the temporal dimension:

$$
g_{\text{EEG}} = \text{MeanPooling}(Z^{\text{EEG}}_{\text{out}}) \in \mathbb{R}^{B \times D}
$$

$$
g_{\text{Eye}} = \text{MeanPooling}(Z^{\text{Eye}}_{\text{out}}) \in \mathbb{R}^{B \times D}
$$

Here, `Z_out^EEG`, `Z_out^Eye ∈ ℝ^{B × T × D}` are the encoder outputs, and the pooling operation averages over the time dimension `T` to produce modality-specific embeddings.

## Feature Fusion

The final multimodal representation is obtained by concatenating the two pooled features:

$$
g_{\text{fused}} = [g_{\text{EEG}} ; g_{\text{Eye}}] \in \mathbb{R}^{B \times 2D}
$$

This fused vector integrates information from both modalities and serves as input to the Domain Adaptation layer.
