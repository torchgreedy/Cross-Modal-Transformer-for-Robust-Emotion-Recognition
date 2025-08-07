# Positional Encoding

## Why Positional Embeddings?

Transformers lack an inherent understanding of sequence order. Unlike RNNs, which process inputs sequentially, Transformers process all timesteps in parallel.

To inject information about the **temporal position** of each input timestep, we introduce **positional encodings**.

This is crucial for time-series data like EEG and eye movement signals, where the position of a signal segment can strongly influence its interpretation.

## Input to Positional Encoding

Before applying positional encoding, both EEG and eye movement features are linearly projected into a common space:

$$
\begin{align*}
\text{EEG} &\rightarrow (B, T, 512) \\
\text{Eye} &\rightarrow (B, T, 512)
\end{align*}
$$

Where:
- $B$ = batch size
- $T$ = sequence length (max = 74 timesteps)
- 512 = model dimension $d_{\text{model}}$

<img src="https://github.com/user-attachments/assets/dba690b6-cb4b-4aae-b090-fc4659fea5b8" alt="Positional Encoding Heatmap" width="600"/>

## Mathematical Formulation (Embedding Step)

Let:
- $Z \in \mathbb{R}^{B \times T \times 512}$ : Projected feature matrix (EEG or Eye)
- $P \in \mathbb{R}^{T \times 512}$ : Positional encoding matrix

Then, we inject position by addition:

$$
Z_{\text{pos}} = Z + P
$$

Where:
- $P_t \in \mathbb{R}^{512}$ is the positional encoding vector for timestep $t$
- $P$ is broadcasted across the batch dimension

## Intuition and Structure

We use sinusoidal positional encoding as introduced in the original Transformer paper:

*Vaswani et al., "Attention is All You Need"*

Each position $t$ in the sequence is represented by a fixed vector that alternates between sine and cosine functions with increasing frequencies.

This design:
- Enables the model to learn relative positions and periodic patterns
- Supports generalization to unseen sequence lengths

## Mathematical Formulation (Encoding Calculation)

For position `pos ∈ [0, T−1]` and dimension `i ∈ [0, d_model / 2 − 1]`:

![Image](https://github.com/user-attachments/assets/6264175d-9cd3-4e28-b1c9-31ab51d83abe)

Even dimensions get sine functions, odd dimensions get cosine functions.
