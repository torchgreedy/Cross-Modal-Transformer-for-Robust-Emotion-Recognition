# Cross-Modal Attention Mechanism

To capture the dependencies between EEG and eye movement signals, we employ a bidirectional cross-modal attention mechanism. Specifically, we allow EEG features to attend to eye movement features and vice versa. This helps the model learn interactions between modalities at each timestep.

## Multi-Head Attention Setup

Let $Z^{\text{EEG}}, Z^{\text{Eye}} \in \mathbb{R}^{B \times T \times D}$ denote the EEG and Eye input features, where $B$ is the batch size, $T$ is the sequence length, and $D$ is the feature dimension. We define $h$ as the number of attention heads, and $d_h = D/h$ as the dimension per head.

These matrices $\bar{A} \in \mathbb{R}^{B \times T \times T}$ represent how timesteps in one modality attend to another and can be visualized as attention heatmaps.

## Linear Projections

We first project the input features into queries, keys, and values:

$$
\begin{aligned}
Q^{\text{EEG}} &= Z^{\text{EEG}} W_Q^{\text{EEG}}, \quad &K^{\text{EEG}} = Z^{\text{EEG}} W_K^{\text{EEG}}, \quad &V^{\text{EEG}} = Z^{\text{EEG}} W_V^{\text{EEG}} \\
Q^{\text{Eye}} &= Z^{\text{Eye}} W_Q^{\text{Eye}}, \quad &K^{\text{Eye}} = Z^{\text{Eye}} W_K^{\text{Eye}}, \quad &V^{\text{Eye}} = Z^{\text{Eye}} W_V^{\text{Eye}}
\end{aligned}
$$

where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ are learnable projection matrices.

## Split into Heads

We reshape these projections into $h$ attention heads:

$$
Q \in \mathbb{R}^{B \times h \times T \times d_h}, \quad K \in \mathbb{R}^{B \times h \times T \times d_h}, \quad V \in \mathbb{R}^{B \times h \times T \times d_h}
$$

## Attention Computation

We compute scaled dot-product attention between the two modalities in both directions.

### EEG → Eye Attention

$$
A^{\text{EEG} \rightarrow \text{Eye}} = \text{softmax} \left( \frac{Q^{\text{EEG}} (K^{\text{Eye}})^\top}{\sqrt{d_h}} + M^{\text{Eye}} \right) \in \mathbb{R}^{B \times h \times T \times T}
$$

Here, $M^{\text{Eye}}$ is the eye attention mask that sets positions corresponding to padded timesteps to $-\infty$.

$$
\text{EEG attended} = A^{\text{EEG} \rightarrow \text{Eye}} \cdot V^{\text{Eye}} \in \mathbb{R}^{B \times h \times T \times d_h}
$$

### Eye → EEG Attention

$$
A^{\text{Eye} \rightarrow \text{EEG}} = \text{softmax} \left( \frac{Q^{\text{Eye}} (K^{\text{EEG}})^\top}{\sqrt{d_h}} + M^{\text{EEG}} \right) \in \mathbb{R}^{B \times h \times T \times T}
$$

Here, $M^{\text{EEG}}$ is the eye attention mask that sets positions corresponding to padded timesteps to $-\infty$.

$$
\text{Eye attended} = A^{\text{Eye} \rightarrow \text{EEG}} \cdot V^{\text{EEG}} \in \mathbb{R}^{B \times h \times T \times d_h}
$$

## Output Projection and Residual Fusion

We concatenate the heads and project back to the original dimension:

$$
\begin{aligned}
\hat{Z}^{\text{EEG}} &= Z^{\text{EEG}} + \text{Proj} \left( \text{ConcatHeads}(\text{EEG attended}) \right) \\
\hat{Z}^{\text{Eye}} &= Z^{\text{Eye}} + \text{Proj} \left( \text{ConcatHeads}(\text{Eye attended}) \right)
\end{aligned}
$$

Here, $\text{Proj}(\cdot)$ denotes a linear transformation back to $\mathbb{R}^{B \times T \times D}$, and residual connections are used to preserve the original modality features.

## Interpretability

For interpretability, we average the attention weights across heads:

$$
\bar{A}^{\text{EEG} \rightarrow \text{Eye}} = \frac{1}{h} \sum_{i=1}^h A_i^{\text{EEG} \rightarrow \text{Eye}}, \quad 
\bar{A}^{\text{Eye} \rightarrow \text{EEG}} = \frac{1}{h} \sum_{i=1}^h A_i^{\text{Eye} \rightarrow \text{EEG}}
$$

These matrices $\bar{A} \in \mathbb{R}^{B \times T \times T}$ represent how timesteps in one modality attend to another and can be visualized as attention heatmaps.

![Cross-modal attention heatmaps. Left: EEG attending to Eye. Right: Eye attending to EEG.](https://github.com/user-attachments/assets/92bd891f-c694-4b12-aaf1-5d569844b898)

<div align="center">Cross-modal attention heatmaps. Left: EEG attending to Eye. Right: Eye attending to EEG</div>
