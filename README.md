#  Cross-Modal Transformer for Emotion Recognition

**Multimodal Architecture Using EEG and Eye-Tracking Features (SEED-V)**
## Model Architecture

![Cross-Modal Transformer Architecture](https://github.com/user-attachments/assets/fd345259-9921-49fa-aa3f-5948e64fab27)

*Model architecture diagram by torchgreedy (Last updated: 2025-08-07)*

![Python](https://img.shields.io/badge/python-3.10-blue) 
![License](https://img.shields.io/badge/license-MIT-green)

##  Overview

This project presents a **Transformer-based multimodal architecture** for classifying human emotions using EEG and eye-tracking signals from the [SEED-V dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html).  
It incorporates **novel attention mechanisms**, **domain adaptation**, and **interpretable gating** to outperform previous benchmarks under rigorous evaluation.

### Core Features

- **Modality-Specific Linear Projections**
- **Bidirectional Cross-Modal Attention**
- **Modality Importance Weighting via Gating Units**
- **Domain Adaptation with Gradient Reversal Layer (GRL)**
- **Subject-Specific Normalization**
- **LOSO Cross-Validation (Leave-One-Subject-Out)**

> **Achieved Accuracy: 75.42% on SEED-V (LOSO evaluation)**

## 📁 Project Structure

```
├── Cross-Modal-Transformer-for-Robust-Emotion-Recognition/
│   ├── checkpoints/                    # 16 model files
│   │   ├── model_fold_1.pth            
│   │   ├── model_fold_2.pth
│   │   ├── ...
│   │   └── model_fold_16.pth
│   ├── SEED-V/
│   │   ├── EEG_DE_features/            # EEG differential entropy features data, just refernce only
│   │   └── Eye_movement_features/      # Eye movement features data, just refernce only
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
