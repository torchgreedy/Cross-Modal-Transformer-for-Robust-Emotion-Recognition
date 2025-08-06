# Cross-Modal-Transformer-for-Robust-Emotion-Recognition

# 🧠 Cross-Modal Transformer for Emotion Recognition  
**Multimodal Architecture Using EEG and Eye-Tracking Features (SEED-V)**

![Python](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🚀 Overview

This project presents a **Transformer-based multimodal architecture** for classifying human emotions using EEG and eye-tracking signals from the [SEED-V dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html).  
It incorporates **novel attention mechanisms**, **domain adaptation**, and **interpretable gating** to outperform previous benchmarks under rigorous evaluation.

### Core Features

-  **Modality-Specific Linear Projections**
-  **Bidirectional Cross-Modal Attention**
-  **Modality Importance Weighting via Gating Units**
-  **Domain Adaptation with Gradient Reversal Layer (GRL)**
-  **Subject-Specific Normalization**
-  **LOSO Cross-Validation (Leave-One-Subject-Out)**

>  **Achieved Accuracy: 75.42% on SEED-V (LOSO evaluation)**

---

## 📁 Project Structure


├── src/
│ ├── model.py # Main multimodal Transformer model
│ ├── modules/
│ │ ├── projections.py # Modality projection layers
│ │ ├── attention.py # Cross-modal and self-attention modules
│ │ ├── grl.py # Domain adaptation (Gradient Reversal Layer)
│ │ ├── fusion.py # Gating and final fusion logic
│ ├── dataset.py # SEED-V preprocessing & Dataloader
│ └── utils.py # Masking, normalization, evaluation helpers
├── train.py # Training script with LOSO evaluation
├── evaluate.py # Standalone evaluation
├── config.yaml # Model and training hyperparameters
├── requirements.txt
└── README.md



