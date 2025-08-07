# Cross-Modal Transformer for Robust Emotion Recognition

*Author: Xuan Truong Nguyen*  
*August 2025*

## Abstract

Emotion recognition from physiological signals is critical for developing human-centered artificial intelligence. In this work, we propose a novel multimodal architecture that fuses Electroencephalogram (EEG) and eye-tracking signals using a Transformer-based model for subject-independent emotion recognition. Our method leverages modality-specific Transformer encoders, bidirectional cross-modal attention, and modality importance weighting to extract and align temporal features across modalities. To ensure robust generalization across individuals, we incorporate domain adversarial training with a Gradient Reversal Layer (GRL) and dynamically schedule adversarial strength over training epochs. Evaluated on the SEED-V dataset using leave-one-subject-out (LOSO) cross-validation, our model achieves competitive performance, demonstrating strong cross-subject generalization and interpretable multimodal fusion. Our results highlight the importance of cross-modal alignment, adaptive weighting, and domain-invariant learning for affective computing applications.

## Introduction

Emotion recognition from physiological signals has gained increasing attention in affective computing and human-computer interaction due to its non-intrusive and objective nature. Among the many physiological modalities, **Electroencephalogram (EEG)** and **eye-tracking** signals are particularly promising due to their complementary sensitivity to cognitive and affective states. However, building robust models that generalize across subjects remains a major challenge, given the high inter-subject variability inherent in physiological data.

Recent advances in deep learning and multimodal learning have enabled more effective modeling of temporal and cross-modal dependencies. In particular, **Transformer-based architectures** offer powerful tools for capturing long-range dependencies and aligning information across different modalities. Furthermore, **domain adversarial training** has emerged as a promising strategy to enforce subject-invariant feature representations, improving generalization in real-world settings.

In this paper, we introduce a **Multimodal Transformer** framework for emotion recognition that integrates EEG and eye-tracking signals. Our contributions are three-fold:

1. We design **modality-specific Transformer encoders** followed by **bidirectional cross-modal attention**, allowing fine-grained interaction between EEG and eye features.
2. We introduce **modality importance weighting** to adaptively control each modality's contribution to the fused representation.
3. We employ **domain adaptation via a Gradient Reversal Layer (GRL)**, coupled with a **dynamic adversarial scheduling mechanism**, to learn domain-invariant representations across subjects.

We evaluate our approach on the **SEED-V** dataset using **Leave-One-Subject-Out (LOSO)** cross-validation to rigorously assess cross-subject generalization. Our experiments demonstrate that the proposed method not only improves accuracy over single-modality and naive fusion baselines but also provides interpretable insights into modality interactions and domain robustness.

## Model Architecture Components

Our cross-modal transformer consists of the following key components:

1. [Linear Projection](modules/linear_projection.md): Aligns EEG and eye-tracking inputs to a shared dimensional space
2. [Positional Encoding](modules/positional_encoding.md): Injects temporal information into the transformer
3. [Feature Importance Module](modules/feature_importance.md): Learns modality-specific importance weights
4. [Cross-Modal Attention](modules/cross_modal_attention.md): Enables bidirectional interaction between modalities
5. [Self-Attention Transformer Encoder](modules/self_attention.md): Captures intra-modal temporal dependencies
6. [Global Pooling and Fusion](modules/global_pooling.md): Creates unified multimodal representations
7. [Domain Adaptation Layer](modules/domain_adaptation.md): Ensures generalization across subjects
8. [Classification Head](modules/classification_head.md): Predicts emotion categories

Each component is described in detail in its respective documentation page.
