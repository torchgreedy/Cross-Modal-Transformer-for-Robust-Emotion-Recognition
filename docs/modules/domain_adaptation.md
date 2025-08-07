# Domain Adaptation Layer

In our multimodal Transformer model, the goal is to generalize emotion recognition across different subjects (i.e., domains). Since EEG and Eye Movement features vary significantly between individuals, the domain adaptation layer enables the model to learn subject-invariant representations, improving generalization to unseen users.

## Architecture Overview

The domain adaptation component comprises three key parts:

1. **Feature Extractor** — the main Transformer model including EEG and Eye modality encoders, cross-modal attention, and fusion.  
2. **Emotion Classifier** — predicts the emotion label from the fused representation.  
3. **Domain Classifier** — predicts the subject identity (i.e., domain) from the same fused representation. This component is trained adversarially via a Gradient Reversal Layer (GRL).

## Mathematical Formulation

Let:  
- $g \in \mathbb{R}^{B \times 2D}$: fused representation after global pooling and concatenation.  
- $f_{\text{cls}}(g)$: emotion classifier output.  
- $f_{\text{dom}}(g)$: domain classifier output.  
- $y$: emotion label.  
- $d$: domain label (subject ID).

### 1. Emotion Classification Loss

$$\mathcal{L}_{\text{cls}} = \frac{1}{B} \sum_{i=1}^{B} \text{CrossEntropy}(f_{\text{cls}}(g_i), y_i)$$

### 2. Domain Classification with Gradient Reversal

Before feeding into the domain classifier, $g$ passes through a GRL:

$$g^{\text{rev}} = \text{GRL}(g)$$

During backpropagation, GRL multiplies gradients by $-\alpha$, where $\alpha$ increases during training. The domain classification loss is:

$$\mathcal{L}_{\text{dom}} = \frac{1}{B} \sum_{i=1}^{B} \text{CrossEntropy}(f_{\text{dom}}(g^{\text{rev}}_i), d_i)$$

### 3. Total Loss (Joint Objective)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{dom}}$$

Here, $\lambda$ is a fixed hyperparameter (e.g., 0.1) controlling the relative weight of domain adaptation in the total loss.

## Gradient Reversal Layer (GRL)

The GRL is a special operation with the following behavior:

- **Forward pass:** Identity function — $\text{GRL}(g) = g$  
- **Backward pass:** Gradient is reversed and scaled:  
  $$\frac{\partial \mathcal{L}}{\partial g} \rightarrow -\alpha \cdot \frac{\partial \mathcal{L}}{\partial g}$$

This adversarial mechanism causes the feature extractor to:  
- Minimize $\mathcal{L}_{\text{cls}}$: learn features useful for emotion classification.  
- Maximize $\mathcal{L}_{\text{dom}}$: learn features that confuse the domain classifier, encouraging domain-invariant representations.

## Dynamic $\alpha$ Scheduling

To gradually increase domain-adversarial learning, we schedule the GRL strength $\alpha$ using a sigmoid ramp-up:

$$\alpha(e) = \frac{2}{1 + \exp(-10 \cdot \frac{e}{E})} - 1$$

Where:  
- $e$: current epoch  
- $E$: total number of training epochs

Behavior:  
- **Early epochs:** $\alpha \approx 0$ — GRL has little effect, allowing the model to focus on learning emotion-discriminative features.  
- **Later epochs:** $\alpha \rightarrow 1$ — GRL becomes stronger, encouraging domain-invariant representation learning.

![Dynamic α scheduling during training. As training progresses, α increases smoothly to strengthen the adversarial signal from the GRL.](https://github.com/user-attachments/assets/86bc4609-58f9-4d66-9d44-97e1aa66e22f)

## Subject-Specific Normalization

This handles inter-subject distribution shifts after adversarial training.

Given a feature vector $x \in \mathbb{R}^d$ for subject $i$, the normalized output is:

$$\text{SubjectNorm}^{(i)}(x) = \gamma^{(i)} \cdot \frac{x - \mu}{\sigma} + \beta^{(i)}$$

Where:

$$\begin{aligned}
\mu &= \frac{1}{d} \sum_{j=1}^{d} x_j \\
\sigma &= \sqrt{\frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2 + \epsilon}
\end{aligned}$$

- $d$: dimensionality of the input (e.g., 512)  
- $\gamma^{(i)}, \beta^{(i)} \in \mathbb{R}^d$: learnable parameters for each subject $i$  
- $\epsilon$: small constant for numerical stability
