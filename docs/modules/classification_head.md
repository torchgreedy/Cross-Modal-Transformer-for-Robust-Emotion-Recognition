# Classification Head

The **Classification Head** maps the fused multimodal representation $g \in \mathbb{R}^{B \times 2D}$ to the emotion label space. This is implemented as a multilayer perceptron (MLP) with GELU activations and dropout regularization.

## Architecture Overview

The classification head consists of three fully connected layers:

$$
\begin{aligned}
h_1 &= \text{GELU}(W_1 g + b_1) \quad &\in \mathbb{R}^{B \times 256} \\
h_1' &= \text{Dropout}(h_1) \\
h_2 &= \text{GELU}(W_2 h_1' + b_2) \quad &\in \mathbb{R}^{B \times 128} \\
h_2' &= \text{Dropout}(h_2) \\
\hat{y} &= \text{Softmax}(W_3 h_2' + b_3) \quad &\in \mathbb{R}^{B \times C}
\end{aligned}
$$

where:
- $g \in \mathbb{R}^{B \times 2D}$ is the fused representation (EEG + Eye).
- $B$ is the batch size.
- $D$ is the feature dimension of each modality.
- $C$ is the number of emotion classes.
- $W_i, b_i$ are learnable parameters of the linear layers.
- GELU is the Gaussian Error Linear Unit activation.
- Dropout is applied after each nonlinearity to prevent overfitting.

## PyTorch Implementation

```python
self.classifier = nn.Sequential(
    nn.Linear(d_model*2, 256),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(256, 128),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(128, n_classes)
)
