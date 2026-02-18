# ICPR-Contest-2026  

*Competition: Beyond Visible Spectrum: AI for Agriculture 2026*  
*Task 2: Boosting automatic crop diseases classification using Sentinel satellite data and self-supervised learning (SSL)*  

This repository contains the code and output for the ICPR Contest 2026 competition focused on crop disease classification using Sentinel-2 satellite imagery. The project explores multiple approaches including self-supervised learning with MoCo and the foundation model Prithvi.  


## Overview  
This project tackles the challenge of automatic crop disease classification using Sentinel-2 multispectral satellite data. We employ advanced deep learning techniques including:  
- **Self-Supervised Learning (SSL)** with MoCo v2 for representation learning
- **Foundation Models** leveraging Prithvi geospatial vision transformers
- **Transfer Learning** using pre-trained encoders

## Dataset  
- **Source** https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026p2/data 
- **Unlabeled Data**
  - **Bands:** 12 Sentinel-2 bands (B1-B9, B8A, B11, B12) 
- **Labeled Data:**
  - **Num of samples:** 900 images
  - **Classes:** Aphid, Blast, RPH, Rust
  - **Bands:** 12 Sentinel-2 bands (B1-B9, B8A, B11, B12) 

## Project Structure
```
ICPR-Contest-2026/
├── classification.ipynb          # Main classification pipeline with MoCo + ResNet50
├── classification_Prithvi.ipynb  # Prithvi foundation model approach
├── ssl_moco_py.ipynb            # Self-supervised learning with MoCo v2
├── pyproject.toml               # Project dependencies
├── output/                      # Model checkpoints and results
└── README.md                    
```

## Approaches  
### 1. Self-Supervised Learning (SSL) with MoCo  
**File:** `ssl_moco_py.ipynb`  
Implementation of Momentum Contrast (MoCo) v2 for self-supervised pre-training on unlabeled Sentinel-2 data.  
- **Architecture:** ResNet50 backbone 
- **Pre-training:** MoCo v2 with momentum encoder
- **Input:** 13-band Sentinel-2 
- **Augmentations:** Random resize, flips, Gaussian blur, brightness adjustment
- **Memory Bank:** 2048 negative samples


### 2. Classification with Custom MoCo Encoder
**File:** `classification.ipynb`  
Fine-tunes a ResNet50 backbone pre-trained with MoCo on the labeled crop disease dataset.  
- **Data Normalization:** Band-specific mean/std computed from data
- **Input:** 13-band Sentinel-2 
- **Task:** Classification with focal loss
- **Metrics:** Macro F1
- **Hyperparameter Optimization:** Optuna tuned Parameters: learning rate, batch size, focal gamma, weight decay, label smoothing

### 3. Foundation Model Approach (Prithvi)
**File:** `classification_Prithvi.ipynb`  
Uses the Prithvi geospatial foundation model, a Vision Transformer pre-trained on NASA's HLS dataset.  
- **Model:** Prithvi ViT (300M/600M parameters)
- **Input:** 6-band Sentinel-2 (B2, B3, B4, B5, B6, B7)
- **Framework:** IBM Terratorch for geospatial AI
- **Task:** Classification with focal loss
- **Normalization:** Band-specific mean/std using prithvi mean/std
- **Metrics:** Macro F1

## Results
Best Model Performance:  
**MoCo + ResNet50 (Linear classification):**
- Validation F1 (Macro): **0.678**
- Accuracy: **87.5%**

**Prithvi Foundation Model:**
- Validation F1 (Macro): **0.55**
- Accuracy: **81.5%**


## Acknowledgments 
ICPR Contest 2026 organizers for the dataset
