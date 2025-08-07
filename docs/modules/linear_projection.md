# Linear Projection

## Purpose

Align EEG and eye-tracking inputs to a shared 512D space for cross-modal processing.

## Input Shape

- EEG: $(B \times 74 \times 310)$
- Eye: $(B \times 74 \times 33)$ 

Where:
- $B$ = batch size
- $T$ = time steps, up to $74$

Before feeding into the Transformers model, both modalities have to be projected into the same dimensional model space (in our case, $512$).

## Mathematical Operation

We denote:

- EEG at timestep $t$: $x_{\text{eeg}}[t] \in \mathbb{R}^{310}$
- Eye at timestep $t$: $x_{\text{eye}}[t] \in \mathbb{R}^{33}$

Each timestep vector $x_t$ is passed through:

$$
z_t = W x_t + b
$$

Where:
- $W$ is a learnable weight matrix (shape $[512 \times \mathrm{input\_dim}]$)
- $b$ is a learnable bias vector (shape $[512]$)

Both $W$ and $b$ are initialized (typically randomly) and then updated by the optimizer during training.

## Gradient Computation

During backpropagation, we compute gradients $\frac{\partial \mathcal{L}}{\partial W}$ and $\frac{\partial \mathcal{L}}{\partial b}$ of the loss $\mathcal{L}$ with respect to these parameters.

## Update Step

The optimizer (Adam) uses those gradients to adjust them. For vanilla gradient descent:

$$
W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
$$

where $\eta$ is the learning rate.

Over many batches/iterations, $W$ and $b$ learn to transform the raw input into useful representations.

## Full Linear Projection

### Projected EEG

$$
z_{\text{eeg}}[t] = W_{\text{eeg}} x_{\text{eeg}}[t] + b_{\text{eeg}} \quad \text{where } W_{\text{eeg}} \in \mathbb{R}^{512 \times 310}
$$

### Projected Eye

$$
z_{\text{eye}}[t] = W_{\text{eye}} x_{\text{eye}}[t] + b_{\text{eye}} \quad \text{where } W_{\text{eye}} \in \mathbb{R}^{512 \times 33}
$$

## After Projection

$$
\begin{align*}
\text{EEG} &\rightarrow (B, T, 512) \\
\text{Eye} &\rightarrow (B, T, 512)
\end{align*}
$$

Now both are aligned to the model space — ready for positional encoding.
