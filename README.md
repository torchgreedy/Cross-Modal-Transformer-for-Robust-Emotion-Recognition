#  Cross-Modal Transformer for Emotion Recognition

**Multimodal Architecture Using EEG and Eye-Tracking Features (SEED-V)**
## Model Architecture

![Cross-Modal Transformer Architecture](https://github.com/user-attachments/assets/fd345259-9921-49fa-aa3f-5948e64fab27)

*Model architecture diagram by torchgreedy (Last updated: 2025-08-07)*

![Python](https://img.shields.io/badge/python-3.10-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

##  Overview

This project presents a **Transformer-based multimodal architecture** for classifying human emotions using EEG and eye-tracking signals from the [SEED-V dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html).  
It incorporates **novel attention mechanisms**, **domain adaptation**, and **interpretable gating** to outperform previous benchmarks under rigorous evaluation.

### Core Features

1. **Positional Encoding**: Sinusoidal encodings enable the model to understand temporal relationships in sequential data.

2. **Feature Importance Module (FIM)**: Learns modality-specific importance weights to adaptively emphasize stronger signals.

3. **Cross-Modal Attention**: Bidirectional attention mechanism allowing EEG features to attend to eye movement features and vice versa.

4. **Self-Attention Transformer Encoder**: Processes each modality to capture intra-modal temporal dynamics.

5. **Global Pooling and Fusion**: Integrates information from both modalities into a unified representation.

6. **Domain Adaptation Layer**: Employs gradient reversal to learn subject-invariant representations, improving generalization.

7. **Classification Head**: Multi-layer perceptron with GELU activations for final emotion classification.

For detailed mathematical formulations and implementation specifics, see our [technical documentation](docs/technical_overview.md).

> **Achieved Accuracy: 75.42% on SEED-V (LOSO evaluation)**
## Model Performance

Our Cross-Modal Transformer achieves **75.42% ± 11.15%** mean accuracy on the SEED-V dataset using rigorous Leave-One-Subject-Out (LOSO) cross-validation.

### Per-Subject Performance

| Subject ID | Accuracy |
|------------|----------|
| Subject 1  | 88.89%   |
| Subject 2  | 84.44%   |
| Subject 3  | 82.22%   |
| Subject 4  | 60.00%   |
| Subject 5  | 75.56%   |
| Subject 6  | 84.44%   |
| Subject 7  | 64.44%   |
| Subject 8  | 86.67%   |
| Subject 9  | 55.56%   |
| Subject 10 | 64.44%   |
| Subject 11 | 75.56%   |
| Subject 12 | 75.56%   |
| Subject 13 | 80.00%   |
| Subject 14 | 57.78%   |
| Subject 15 | 80.00%   |
| Subject 16 | 91.11%   |
| **Average**| **75.42%**|

The variance in performance across subjects (standard deviation: 11.15%) highlights the challenge of cross-subject generalization in physiological emotion recognition, which our domain adaptation approach helps address.

## 📁 Project Structure

```
├── Cross-Modal-Transformer-for-Robust-Emotion-Recognition/
│   ├── checkpoints/                    # 16 model files
│   │   ├── model_fold_1.pth            
│   │   ├── model_fold_2.pth
│   │   ├── ...
│   │   └── model_fold_16.pth
│   ├── docs/                           # Technical documentation
│   │   ├── technical_overview.md       # Overall architecture overview
│   │   └── modules/                    # Detailed module descriptions
│   │       ├── classification_head.md  # Classification component details
│   │       ├── cross_modal_attention.md# Bidirectional attention mechanism
│   │       ├── domain_adaptation.md    # Subject-invariant learning approach
│   │       ├── feature_importance.md   # Adaptive modality weighting
│   │       ├── global_pooling.md       # Feature fusion methods
│   │       ├── linear_projection.md    # Input dimension alignment
│   │       ├── positional_encoding.md  # Temporal information encoding
│   │       └── self_attention.md       # Intra-modal attention mechanism
│   ├── SEED-V/
│   │   ├── EEG_DE_features/            # EEG differential entropy features data, just reference only
│   │   └── Eye_movement_features/      # Eye movement features data, just reference only
│   ├── interpretability_results/
│   │   ├── attention_analysis/
│   │   │   ├── attention_heatmaps.png
│   │   │   └── temporal_attention_patterns.png
│   │   ├── confidence_analysis/
│   │   │   └── confidence_analysis.png
│   │   ├── emotion_patterns/
│   │   │   └── emotion_specific_attention.png
│   │   ├── subject_analysis/
│   │   │   └── subject_variability_analysis.png
│   │   └── confusion_matrix.png
│   ├── src/
│   │   ├── model.py                    # Main multimodal Transformer model
│   │   ├── modules/
│   │   │   ├── projections.py          # Modality projection layers
│   │   │   ├── attention.py            # Cross-modal and self-attention modules
│   │   │   ├── grl.py                  # Domain adaptation (Gradient Reversal Layer)
│   │   │   ├── fusion.py               # Gating and final fusion logic
│   │   │   └── __init__.py             # Module exports
│   │   ├── dataset.py                  # SEED-V preprocessing & Dataloader
│   │   ├── utils.py                    # Masking, normalization, evaluation helpers
│   │   └── __init__.py                 # Package exports
│   ├── train.py                        # Training script with LOSO evaluation
│   ├── evaluate.py                     # Standalone evaluation
│   ├── config.yaml                     # Model and training hyperparameters
│   ├── requirements.txt                # Project dependencies
│   ├── LICENSE                         # Project license
│   └── README.md                       # Project documentation
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/torchgreedy/Cross-Modal-Transformer-for-Robust-Emotion-Recognition.git
cd Cross-Modal-Transformer-for-Robust-Emotion-Recognition
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

##  Dataset

This project uses the SEED-V dataset containing EEG and eye-tracking data for emotion recognition. The dataset includes:

- **5 emotion categories**: happy, sad, fear, disgust, neutral
- **16 subjects** with 3 sessions each
- **62-channel EEG data** and **eye movement features**

You'll need to request access to the dataset from the [BCMI lab](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html).

##  Usage

### Training

To train the model with Leave-One-Subject-Out cross-validation:

```bash
python train.py --config config.yaml
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model_path checkpoints/model_fold_1.pth --subject_id 1
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
