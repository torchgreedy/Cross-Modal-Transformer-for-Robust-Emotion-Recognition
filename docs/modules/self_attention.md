# Self-Attention Transformer Encoder

After cross-modal interaction, we apply a Transformer Encoder to each modality to enable intra-modal contextual modeling across time steps. Each encoder layer consists of a Multi-Head Self-Attention block and a Position-wise Feedforward Network, each wrapped in residual connections and layer normalization.

## Multi-Head Self-Attention

Let the input be $Z \in \mathbb{R}^{B \times T \times D}$, where:
- $B$: batch size
- $T$: number of timesteps
- $D$: hidden dimension

For each attention head $i \in \{1, \dots, h\}$, we compute:

$$
\begin{aligned}
Q_i &= Z W^Q_i \in \mathbb{R}^{B \times T \times d_h} \\
K_i &= Z W^K_i \in \mathbb{R}^{B \times T \times d_h} \\
V_i &= Z W^V_i \in \mathbb{R}^{B \times T \times d_h}
\end{aligned}
$$

where:
- $d_h = \frac{D}{h}$: dimension per head
- $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{D \times d_h}$: learnable projections

We then compute scaled dot-product attention:

$$
\text{Attention}_i = \text{softmax}\left( \frac{Q_i K_i^\top}{\sqrt{d_h}} + M \right) V_i
$$

Here, $M \in \mathbb{R}^{B \times T \times T}$ is an optional attention mask, with values set to $-\infty$ for padded positions and $0$ elsewhere.

The multi-head attention output is computed by concatenating all heads and applying a final linear projection:

$$
\text{MultiHead}(Z) = \text{Concat}(\text{Attention}_1, \dots, \text{Attention}_h) W^O \in \mathbb{R}^{B \times T \times D}
$$

## Position-wise Feedforward Network

A two-layer feedforward network is applied independently to each timestep:

$$
\text{FFN}(x) = \text{Dropout}(W_2 (\text{GELU}(W_1 x)) + b_2) \in \mathbb{R}^{B \times T \times D}
$$

where:
- $W_1 \in \mathbb{R}^{D \times D_{ff}}$, $W_2 \in \mathbb{R}^{D_{ff} \times D}$
- $D_{ff}$: inner hidden size (e.g., 1024)

## Final Encoder Output

Residual connections and layer normalization are applied as follows:

$$
\begin{aligned}
Z' &= \text{LayerNorm}(Z + \text{Dropout}(\text{MultiHead}(Z))) \\
Z_{\text{out}} &= \text{LayerNorm}(Z' + \text{FFN}(Z'))
\end{aligned}
$$

The resulting $Z_{\text{out}} \in \mathbb{R}^{B \times T \times D}$ is forwarded to Global Pooling and Fusion stage.
