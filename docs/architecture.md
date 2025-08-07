\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{float}

\title{Cross-Modal Transformer for Robust Emotion Recognition}
\author{Xuan Truong Nguyen}
\date{August 2025}

\begin{document}

\maketitle

\vspace{1\baselineskip}  % space after title
\begin{abstract}
Emotion recognition from physiological signals is critical for developing human-centered artificial intelligence. In this work, we propose a novel multimodal architecture that fuses Electroencephalogram (EEG) and eye-tracking signals using a Transformer-based model for subject-independent emotion recognition. Our method leverages modality-specific Transformer encoders, bidirectional cross-modal attention, and modality importance weighting to extract and align temporal features across modalities. To ensure robust generalization across individuals, we incorporate domain adversarial training with a Gradient Reversal Layer (GRL) and dynamically schedule adversarial strength over training epochs. Evaluated on the SEED-V dataset using leave-one-subject-out (LOSO) cross-validation, our model achieves competitive performance, demonstrating strong cross-subject generalization and interpretable multimodal fusion. Our results highlight the importance of cross-modal alignment, adaptive weighting, and domain-invariant learning for affective computing applications.
\end{abstract}

\section{Introduction}

Emotion recognition from physiological signals has gained increasing attention in affective computing and human-computer interaction due to its non-intrusive and objective nature. Among the many physiological modalities, \textbf{Electroencephalogram (EEG)} and \textbf{eye-tracking} signals are particularly promising due to their complementary sensitivity to cognitive and affective states. However, building robust models that generalize across subjects remains a major challenge, given the high inter-subject variability inherent in physiological data.

Recent advances in deep learning and multimodal learning have enabled more effective modeling of temporal and cross-modal dependencies. In particular, \textbf{Transformer-based architectures} offer powerful tools for capturing long-range dependencies and aligning information across different modalities. Furthermore, \textbf{domain adversarial training} has emerged as a promising strategy to enforce subject-invariant feature representations, improving generalization in real-world settings.

In this paper, we introduce a \textbf{Multimodal Transformer} framework for emotion recognition that integrates EEG and eye-tracking signals. Our contributions are three-fold:
\begin{enumerate}
    \item We design \textbf{modality-specific Transformer encoders} followed by \textbf{bidirectional cross-modal attention}, allowing fine-grained interaction between EEG and eye features.
    \item We introduce \textbf{modality importance weighting} to adaptively control each modality’s contribution to the fused representation.
    \item We employ \textbf{domain adaptation via a Gradient Reversal Layer (GRL)}, coupled with a \textbf{dynamic adversarial scheduling mechanism}, to learn domain-invariant representations across subjects.
\end{enumerate}

We evaluate our approach on the \textbf{SEED-V} dataset using \textbf{Leave-One-Subject-Out (LOSO)} cross-validation to rigorously assess cross-subject generalization. Our experiments demonstrate that the proposed method not only improves accuracy over single-modality and naive fusion baselines but also provides interpretable insights into modality interactions and domain robustness.

\section{Linear Projection}

\vspace{1\baselineskip}  % space before section content

\textbf{Purpose: }Align EEG and eye-tracking inputs to a shared 512D space for cross-modal processing.

\vspace{0.5\baselineskip}

\textbf{Input shape:}

EEG: $(B \times 74 \times 310)$

Eye: $(B \times 74 \times 33)$ \quad ( $B$ = batch size, $T$ = time steps, up to $74$)

Before feeding into the Transformers model, both modalities have to be projected into the same dimensional model space (in our case, $512$).

\vspace{0.5\baselineskip}

\textbf{Mathematical Operation:}

We denote:

\quad EEG at timestep $t$: $x_{\text{eeg}}[t] \in \mathbb{R}^{310}$

\quad Eye at timestep $t$: $x_{\text{eye}}[t] \in \mathbb{R}^{33}$

Each timestep vector $x_t$ is passed through:

\[
z_t = W x_t + b
\]

Where:

\quad $W$ is a learnable weight matrix (shape $[512 \times \text{input\_dim}]$)

\quad $b$ is a learnable bias vector (shape $[512]$)

Both $W$ and $b$ are initialized (typically randomly) and then updated by the optimizer during training.

\vspace{0.5\baselineskip}

\textbf{Gradient computation:} During backpropagation, we compute gradients $\frac{\partial \mathcal{L}}{\partial W}$ and $\frac{\partial \mathcal{L}}{\partial b}$ of the loss $\mathcal{L}$ with respect to these parameters.

\textbf{Update step:} The optimizer (Adam) uses those gradients to adjust them. For vanilla gradient descent:

\[
W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
\]

where $\eta$ is the learning rate.

Over many batches/iterations, $W$ and $b$ learn to transform the raw input into useful representations.

\vspace{0.5\baselineskip}

\textbf{Full Linear Projection:}

Projected EEG:

\[
z_{\text{eeg}}[t] = W_{\text{eeg}} x_{\text{eeg}}[t] + b_{\text{eeg}} \quad \text{where } W_{\text{eeg}} \in \mathbb{R}^{512 \times 310}
\]

Projected Eye:

\[
z_{\text{eye}}[t] = W_{\text{eye}} x_{\text{eye}}[t] + b_{\text{eye}} \quad \text{where } W_{\text{eye}} \in \mathbb{R}^{512 \times 33}
\]

\textbf{After projection:}

\begin{align*}
\text{EEG} &\rightarrow (B, T, 512) \\
\text{Eye} &\rightarrow (B, T, 512)
\end{align*}

Now both are aligned to the model space — ready for positional encoding.

\vspace{2\baselineskip}  % extra space before next section

\section{Positional Encoding}

\vspace{1\baselineskip}

\subsection{Why Positional Embeddings?}

Transformers lack an inherent understanding of sequence order. Unlike RNNs, which process inputs sequentially, Transformers process all timesteps in parallel.

To inject information about the \textbf{temporal position} of each input timestep, we introduce \textbf{positional encodings}.

This is crucial for time-series data like EEG and eye movement signals, where the position of a signal segment can strongly influence its interpretation.

\vspace{0.5\baselineskip}

\subsection{Input to Positional Encoding}

Before applying positional encoding, both EEG and eye movement features are linearly projected into a common space:

\begin{align*}
\text{EEG} &\rightarrow (B, T, 512) \\
\text{Eye} &\rightarrow (B, T, 512)
\end{align*}

Where:

\quad $B$ = batch size

\quad $T$ = sequence length (max = 74 timesteps)

\quad 512 = model dimension $d_{\text{model}}$

\vspace{1\baselineskip}  % space before figure
\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{positional_encoding_heatmap.png}
    \caption{A heatmap of sinusoidal positional encodings across 74 timesteps and 512 dimensions.}
    \label{fig:positional_encoding}
\end{figure}
\vspace{1\baselineskip}  % space after figure

\subsection{Mathematical Formulation (Embedding Step)}

Let:

\begin{itemize}
  \item \quad $Z \in \mathbb{R}^{B \times T \times 512}$ : Projected feature matrix (EEG or Eye)
  \item \quad $P \in \mathbb{R}^{T \times 512}$ : Positional encoding matrix
\end{itemize}

Then, we inject position by addition:

\[
Z_{\text{pos}} = Z + P
\]

Where:

\begin{itemize}
  \item \quad $P_t \in \mathbb{R}^{512}$ is the positional encoding vector for timestep $t$
  \item \quad $P$ is broadcasted across the batch dimension
\end{itemize}

\vspace{0.5\baselineskip}

\subsection{Intuition and Structure}

We use sinusoidal positional encoding as introduced in the original Transformer paper:

\textit{Vaswani et al., "Attention is All You Need"}

Each position $t$ in the sequence is represented by a fixed vector that alternates between sine and cosine functions with increasing frequencies.

This design:

\begin{itemize}
  \item \quad Enables the model to learn relative positions and periodic patterns
  \item \quad Supports generalization to unseen sequence lengths
\end{itemize}

\vspace{0.5\baselineskip}

\subsection{Mathematical Formulation (Encoding Calculation)}

For position $pos \in [0, T-1]$ and dimension $i \in [0, d_{\text{model}}/2 - 1]$:

\[
\text{PE}_{pos, 2i} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
\]
\[
\text{PE}_{pos, 2i+1} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
\]

Where:

\begin{itemize}
  \item \quad $d_{\text{model}} = 512$ (in our case)
  \item \quad $\text{PE}_{pos} \in \mathbb{R}^{512}$ is the encoding for timestep $pos$
\end{itemize}

Even dimensions get sine functions, odd dimensions get cosine functions.

\section{Feature Importance Module (FIM)}

\subsection{Purpose}
The Feature Importance Module is designed to learn modality-specific importance weights that indicate how relevant each modality is for emotion recognition adaptively across different samples, timesteps, and feature dimensions. This helps:

\begin{itemize}
  \item Emphasize stronger signals (e.g., if EEG is more informative for a sample, weight it more)
  \item Reduce reliance on noisy or less relevant modalities
\end{itemize}

\subsection{Architectural Role}
Assume that after linear projection and positional encoding, we obtain:
\begin{align*}
    Z_{\text{eeg}} &\in \mathbb{R}^{B \times T \times d} \\
    Z_{\text{eye}} &\in \mathbb{R}^{B \times T \times d}
\end{align*}
These represent the EEG and eye-tracking features, respectively, where:
\begin{itemize}
  \item $B$: Batch size
  \item $T$: Sequence length (timesteps)
  \item $d$: Feature dimension after projection
\end{itemize}

These modality-specific features are processed independently at first. Before fusion or cross-modal attention, we apply the Feature Importance Module to reweight each feature tensor.

\subsection{Mathematical Formulation}

\vspace{1\baselineskip}

\textbf{1. Modality-Specific Importance Estimation}

We use a small feedforward network to compute a scalar importance score for each modality at each timestep.

Let:

\begin{itemize}
  \item \quad $W_f \in \mathbb{R}^{d \times 1}$, \quad $b_f \in \mathbb{R}$
\end{itemize}

Then, for each timestep $t \in [1, T]$, compute:

\[
\alpha_{\text{eeg}}[t] = \sigma(Z_{\text{eeg}}[t] W_f + b_f) \in \mathbb{R}
\]
\[
\alpha_{\text{eye}}[t] = \sigma(Z_{\text{eye}}[t] W_f + b_f) \in \mathbb{R}
\]

where $\sigma(\cdot)$ denotes the sigmoid activation function to squash values to the range $[0, 1]$.

\vspace{1\baselineskip}

\textbf{2. Reweighting the Modalities}

We then reweight each modality’s features using the learned importance scores:

\[
\tilde{Z}_{\text{eeg}}[t] = \alpha_{\text{eeg}}[t] \cdot Z_{\text{eeg}}[t]
\]
\[
\tilde{Z}_{\text{eye}}[t] = \alpha_{\text{eye}}[t] \cdot Z_{\text{eye}}[t]
\]

These rescaled embeddings $\tilde{Z}_{\text{eeg}}$ and $\tilde{Z}_{\text{eye}}$ are passed to the subsequent fusion or attention modules in the model.


\section{Cross-Modal Attention Mechanism}

To capture the dependencies between EEG and eye movement signals, we employ a bidirectional cross-modal attention mechanism. Specifically, we allow EEG features to attend to eye movement features and vice versa. This helps the model learn interactions between modalities at each timestep.

\subsection{Multi-Head Attention Setup}

Let $Z^{\text{EEG}}, Z^{\text{Eye}} \in \mathbb{R}^{B \times T \times D}$ denote the EEG and Eye input features, where $B$ is the batch size, $T$ is the sequence length, and $D$ is the feature dimension. We define $h$ as the number of attention heads, and $d_h = D/h$ as the dimension per head.

These matrices $\bar{A} \in \mathbb{R}^{B \times T \times T}$ represent how timesteps in one modality attend to another and can be visualized as attention heatmaps.


\subsubsection*{Linear Projections}

We first project the input features into queries, keys, and values:

\[
\begin{aligned}
Q^{\text{EEG}} &= Z^{\text{EEG}} W_Q^{\text{EEG}}, \quad &K^{\text{EEG}} = Z^{\text{EEG}} W_K^{\text{EEG}}, \quad &V^{\text{EEG}} = Z^{\text{EEG}} W_V^{\text{EEG}} \\
Q^{\text{Eye}} &= Z^{\text{Eye}} W_Q^{\text{Eye}}, \quad &K^{\text{Eye}} = Z^{\text{Eye}} W_K^{\text{Eye}}, \quad &V^{\text{Eye}} = Z^{\text{Eye}} W_V^{\text{Eye}}
\end{aligned}
\]

where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ are learnable projection matrices.

\subsubsection*{Split into Heads}

We reshape these projections into $h$ attention heads:

\[
Q \in \mathbb{R}^{B \times h \times T \times d_h}, \quad K \in \mathbb{R}^{B \times h \times T \times d_h}, \quad V \in \mathbb{R}^{B \times h \times T \times d_h}
\]

\subsection{Attention Computation}

We compute scaled dot-product attention between the two modalities in both directions.

\subsubsection*{EEG $\rightarrow$ Eye Attention}

\[
A^{\text{EEG} \rightarrow \text{Eye}} = \text{softmax} \left( \frac{Q^{\text{EEG}} (K^{\text{Eye}})^\top}{\sqrt{d_h}} + M^{\text{Eye}} \right) \in \mathbb{R}^{B \times h \times T \times T}
\]

Here, $M^{\text{Eye}}$ is the eye attention mask that sets positions corresponding to padded timesteps to $-\infty$.

\[
\text{EEG attended} = A^{\text{EEG} \rightarrow \text{Eye}} \cdot V^{\text{Eye}} \in \mathbb{R}^{B \times h \times T \times d_h}
\]

\subsubsection*{Eye $\rightarrow$ EEG Attention}

\[
A^{\text{Eye} \rightarrow \text{EEG}} = \text{softmax} \left( \frac{Q^{\text{Eye}} (K^{\text{EEG}})^\top}{\sqrt{d_h}} + M^{\text{EEG}} \right) \in \mathbb{R}^{B \times h \times T \times T}
\]

Here, $M^{\text{EEG}}$ is the eye attention mask that sets positions corresponding to padded timesteps to $-\infty$.

\[
\text{Eye attended} = A^{\text{Eye} \rightarrow \text{EEG}} \cdot V^{\text{EEG}} \in \mathbb{R}^{B \times h \times T \times d_h}
\]

\subsection{Output Projection and Residual Fusion}

We concatenate the heads and project back to the original dimension:

\[
\begin{aligned}
\hat{Z}^{\text{EEG}} &= Z^{\text{EEG}} + \text{Proj} \left( \text{ConcatHeads}(\text{EEG attended}) \right) \\
\hat{Z}^{\text{Eye}} &= Z^{\text{Eye}} + \text{Proj} \left( \text{ConcatHeads}(\text{Eye attended}) \right)
\end{aligned}
\]

Here, $\text{Proj}(\cdot)$ denotes a linear transformation back to $\mathbb{R}^{B \times T \times D}$, and residual connections are used to preserve the original modality features.

\subsection{Interpretability}

For interpretability, we average the attention weights across heads:

\[
\bar{A}^{\text{EEG} \rightarrow \text{Eye}} = \frac{1}{h} \sum_{i=1}^h A_i^{\text{EEG} \rightarrow \text{Eye}}, \quad 
\bar{A}^{\text{Eye} \rightarrow \text{EEG}} = \frac{1}{h} \sum_{i=1}^h A_i^{\text{Eye} \rightarrow \text{EEG}}
\]

These matrices $\bar{A} \in \mathbb{R}^{B \times T \times T}$ represent how timesteps in one modality attend to another and can be visualized as attention heatmaps.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\textwidth]{attention_heatmaps.png}
    \caption{Cross-modal attention heatmaps. Left: EEG attending to Eye. Right: Eye attending to EEG.}
    \label{fig:cross-modal-attn}
\end{figure}

\section{Self-Attention Transformer Encoder}

After cross-modal interaction, we apply a Transformer Encoder to each modality to enable intra-modal contextual modeling across time steps. Each encoder layer consists of a Multi-Head Self-Attention block and a Position-wise Feedforward Network, each wrapped in residual connections and layer normalization.

\subsubsection*{Multi-Head Self-Attention}

Let the input be $Z \in \mathbb{R}^{B \times T \times D}$, where:
\begin{itemize}
  \item $B$: batch size
  \item $T$: number of timesteps
  \item $D$: hidden dimension
\end{itemize}

For each attention head $i \in \{1, \dots, h\}$, we compute:
\[
\begin{aligned}
Q_i &= Z W^Q_i \in \mathbb{R}^{B \times T \times d_h} \\
K_i &= Z W^K_i \in \mathbb{R}^{B \times T \times d_h} \\
V_i &= Z W^V_i \in \mathbb{R}^{B \times T \times d_h}
\end{aligned}
\]
where:
\begin{itemize}
  \item $d_h = \frac{D}{h}$: dimension per head
  \item $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{D \times d_h}$: learnable projections
\end{itemize}

We then compute scaled dot-product attention:
\[
\text{Attention}_i = \text{softmax}\left( \frac{Q_i K_i^\top}{\sqrt{d_h}} + M \right) V_i
\]
Here, $M \in \mathbb{R}^{B \times T \times T}$ is an optional attention mask, with values set to $-\infty$ for padded positions and $0$ elsewhere.

The multi-head attention output is computed by concatenating all heads and applying a final linear projection:
\[
\text{MultiHead}(Z) = \text{Concat}(\text{Attention}_1, \dots, \text{Attention}_h) W^O \in \mathbb{R}^{B \times T \times D}
\]

\subsubsection*{Position-wise Feedforward Network}

A two-layer feedforward network is applied independently to each timestep:
\[
\text{FFN}(x) = \text{Dropout}(W_2 (\text{GELU}(W_1 x)) + b_2) \in \mathbb{R}^{B \times T \times D}
\]
where:
\begin{itemize}
  \item $W_1 \in \mathbb{R}^{D \times D_{ff}}$, $W_2 \in \mathbb{R}^{D_{ff} \times D}$
  \item $D_{ff}$: inner hidden size (e.g., 1024)
\end{itemize}

\subsubsection*{Final Encoder Output}

Residual connections and layer normalization are applied as follows:
\[
\begin{aligned}
Z' &= \text{LayerNorm}(Z + \text{Dropout}(\text{MultiHead}(Z))) \\
Z_{\text{out}} &= \text{LayerNorm}(Z' + \text{FFN}(Z'))
\end{aligned}
\]

The resulting $Z_{\text{out}} \in \mathbb{R}^{B \times T \times D}$ is forwarded to Global Pooling and Fusion stage.

\section{Global Pooling and Fusion}
\subsection{Global Pooling}
After modality-specific Transformer encoders, we extract a fixed-size representation for each modality by applying global average pooling across the temporal dimension:

\[
\begin{aligned}
g_{\text{EEG}} &= \text{MeanPooling}(Z^{\text{EEG}}_{\text{out}}) \in \mathbb{R}^{B \times D} \\
g_{\text{Eye}} &= \text{MeanPooling}(Z^{\text{Eye}}_{\text{out}}) \in \mathbb{R}^{B \times D}
\end{aligned}
\]

Here, $Z^{\text{EEG}}_{\text{out}}, Z^{\text{Eye}}_{\text{out}} \in \mathbb{R}^{B \times T \times D}$ are the encoder outputs, and the pooling operation averages over the time dimension $T$ to produce modality-specific embeddings.

\subsection{Feature Fusion}

The final multimodal representation is obtained by concatenating the two pooled features:

\[
g_{\text{fused}} = [g_{\text{EEG}} ; g_{\text{Eye}}] \in \mathbb{R}^{B \times 2D}
\]

This fused vector integrates information from both modalities and serves as input to the Domain Adaptation layer.

\section{Domain Adaptation Layer}

In our multimodal Transformer model, the goal is to generalize emotion recognition across different subjects (i.e., domains). Since EEG and Eye Movement features vary significantly between individuals, the domain adaptation layer enables the model to learn subject-invariant representations, improving generalization to unseen users.

\subsection{Architecture Overview}

The domain adaptation component comprises three key parts:

\begin{enumerate}
    \item \textbf{Feature Extractor} — the main Transformer model including EEG and Eye modality encoders, cross-modal attention, and fusion.
    \item \textbf{Emotion Classifier} — predicts the emotion label from the fused representation.
    \item \textbf{Domain Classifier} — predicts the subject identity (i.e., domain) from the same fused representation. This component is trained adversarially via a Gradient Reversal Layer (GRL).
\end{enumerate}

\subsection{Mathematical Formulation}

Let:
\begin{itemize}
    \item \( g \in \mathbb{R}^{B \times 2D} \): fused representation after global pooling and concatenation.
    \item \( f_{\text{cls}}(g) \): emotion classifier output.
    \item \( f_{\text{dom}}(g) \): domain classifier output.
    \item \( y \): emotion label.
    \item \( d \): domain label (subject ID).
\end{itemize}

\paragraph{1. Emotion Classification Loss}
\[
\mathcal{L}_{\text{cls}} = \frac{1}{B} \sum_{i=1}^{B} \text{CrossEntropy}(f_{\text{cls}}(g_i), y_i)
\]

\paragraph{2. Domain Classification with Gradient Reversal}
Before feeding into the domain classifier, \( g \) passes through a GRL:
\[
g^{\text{rev}} = \text{GRL}(g)
\]
During backpropagation, GRL multiplies gradients by \( -\alpha \), where \( \alpha \) increases during training. The domain classification loss is:
\[
\mathcal{L}_{\text{dom}} = \frac{1}{B} \sum_{i=1}^{B} \text{CrossEntropy}(f_{\text{dom}}(g^{\text{rev}}_i), d_i)
\]

\paragraph{3. Total Loss (Joint Objective)}
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{dom}}
\]
Here, \( \lambda \) is a fixed hyperparameter (e.g., 0.1) controlling the relative weight of domain adaptation in the total loss.

\subsection{Gradient Reversal Layer (GRL)}

The GRL is a special operation with the following behavior:
\begin{itemize}
    \item \textbf{Forward pass:} Identity function — \( \text{GRL}(g) = g \)
    \item \textbf{Backward pass:} Gradient is reversed and scaled:
    \[
    \frac{\partial \mathcal{L}}{\partial g} \rightarrow -\alpha \cdot \frac{\partial \mathcal{L}}{\partial g}
    \]
\end{itemize}

This adversarial mechanism causes the feature extractor to:
\begin{itemize}
    \item Minimize \( \mathcal{L}_{\text{cls}} \): learn features useful for emotion classification.
    \item Maximize \( \mathcal{L}_{\text{dom}} \): learn features that confuse the domain classifier, encouraging domain-invariant representations.
\end{itemize}

\subsection{Dynamic \(\alpha\) Scheduling}

To gradually increase domain-adversarial learning, we schedule the GRL strength \(\alpha\) using a sigmoid ramp-up:
\[
\alpha(e) = \frac{2}{1 + \exp(-10 \cdot \frac{e}{E})} - 1
\]
Where:
\begin{itemize}
    \item \( e \): current epoch
    \item \( E \): total number of training epochs
\end{itemize}

\noindent Behavior:
\begin{itemize}
    \item \textbf{Early epochs:} \( \alpha \approx 0 \) — GRL has little effect, allowing the model to focus on learning emotion-discriminative features.
    \item \textbf{Later epochs:} \( \alpha \rightarrow 1 \) — GRL becomes stronger, encouraging domain-invariant representation learning.
\end{itemize}


\begin{figure}[h]
    \centering
    \includegraphics[width=0.6\textwidth]{alpha_schedule.png}
    \caption{Dynamic $\alpha$ scheduling during training. As training progresses, $\alpha$ increases smoothly to strengthen the adversarial signal from the GRL.}
    \label{fig:alpha-schedule}
\end{figure}

\subsection{Subject-Specific Normalization}

This handles inter-subject distribution shifts after adversarial training.

Given a feature vector $x \in \mathbb{R}^d$ for subject $i$, the normalized output is:

\[
\text{SubjectNorm}^{(i)}(x) = \gamma^{(i)} \cdot \frac{x - \mu}{\sigma} + \beta^{(i)}
\]

Where:
\begin{align*}
\mu &= \frac{1}{d} \sum_{j=1}^{d} x_j \\
\sigma &= \sqrt{\frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2 + \epsilon}
\end{align*}

\begin{itemize}
    \item $d$: dimensionality of the input (e.g., 512)
    \item $\gamma^{(i)}, \beta^{(i)} \in \mathbb{R}^d$: learnable parameters for each subject $i$
    \item $\epsilon$: small constant for numerical stability
\end{itemize}

\section{Classification Head}

The \textbf{Classification Head} maps the fused multimodal representation $g \in \mathbb{R}^{B \times 2D}$ to the emotion label space. This is implemented as a multilayer perceptron (MLP) with GELU activations and dropout regularization.

\subsection{Architecture Overview}

The classification head consists of three fully connected layers:

\[
\begin{aligned}
h_1 &= \text{GELU}(W_1 g + b_1) \quad &\in \mathbb{R}^{B \times 256} \\
h_1' &= \text{Dropout}(h_1) \\
h_2 &= \text{GELU}(W_2 h_1' + b_2) \quad &\in \mathbb{R}^{B \times 128} \\
h_2' &= \text{Dropout}(h_2) \\
\hat{y} &= \text{Softmax}(W_3 h_2' + b_3) \quad &\in \mathbb{R}^{B \times C}
\end{aligned}
\]

\noindent where:
\begin{itemize}
  \item $g \in \mathbb{R}^{B \times 2D}$ is the fused representation (EEG + Eye).
  \item $B$ is the batch size.
  \item $D$ is the feature dimension of each modality.
  \item $C$ is the number of emotion classes.
  \item $W_i, b_i$ are learnable parameters of the linear layers.
  \item GELU is the Gaussian Error Linear Unit activation.
  \item Dropout is applied after each nonlinearity to prevent overfitting.
\end{itemize}

\subsection{PyTorch Implementation}
\begin{verbatim}
self.classifier = nn.Sequential(
    nn.Linear(d_model*2, 256),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(256, 128),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(128, n_classes)
)
\end{verbatim}

\subsection{Loss Function}

The predicted class probabilities $\hat{y}$ are compared to the ground truth emotion labels $y$ using the standard cross-entropy loss:

\[
\mathcal{L}_{\text{cls}} = \frac{1}{B} \sum_{i=1}^B \text{CrossEntropy}(\hat{y}_i, y_i)
\]

This loss is combined with the domain loss $\mathcal{L}_{\text{dom}}$ during joint training.


\end{document}
